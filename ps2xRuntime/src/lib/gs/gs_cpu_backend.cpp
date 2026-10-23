#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/ps2_gs_common.h"
#include "runtime/gs/ps2_gs_psmct16.h"
#include "runtime/gs/ps2_gs_psmct32.h"
#include "ThreadNaming.h"
#include "runtime/gs/ps2_gs_psmt4.h"
#include "runtime/gs/ps2_gs_psmt8.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "ps2_log.h"
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <set>
#include <string>
#include <tuple>
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

using namespace GSInternal;

namespace
{
    float fabsQ(float q)
    {
        return (std::fabs(q) > 1.0e-8f) ? q : 1.0f;
    }

    u16 Rgba8888ToRgba5551(u32 c)
    {
        uint32_t r = ((c >> 0) & 0xFF) >> 3;
        uint32_t g = ((c >> 8) & 0xFF) >> 3;
        uint32_t b = ((c >> 16) & 0xFF) >> 3;
        uint32_t a = ((c >> 24) & 0xFF) >> 7;

        return (r | (g << 5) | (b << 10) | (a << 15));
    }

    u32 Rgba5551ToRgba8888(u16 c)
    {
        u32 r = ((c >> 0) & 0x1F) << 3;
        u32 g = ((c >> 5) & 0x1F) << 3;
        u32 b = ((c >> 10) & 0x1F) << 3;
        u32 a = ((c >> 15) & 0x01) << 7;

        return (r | (g << 8) | (b << 16) | (a << 24));
    }

    u32 pack32(u8 r, u8 g, u8 b, u8 a)
    {
        return static_cast<u32>(r) | (g << 8) | (b << 16) | (a << 24);
    }

    uint32_t applyTexa(const GSTexaReg &texa, uint8_t psm, uint32_t texel)
    {
        if (psm == GS_PSM_CT32)
            return texel;

        const uint8_t r = static_cast<uint8_t>(texel & 0xFFu);
        const uint8_t g = static_cast<uint8_t>((texel >> 8) & 0xFFu);
        const uint8_t b = static_cast<uint8_t>((texel >> 16) & 0xFFu);
        const bool rgbZero = r == 0u && g == 0u && b == 0u;
        uint8_t a = static_cast<uint8_t>((texel >> 24) & 0xFFu);

        switch (psm)
        {
        case GS_PSM_CT24:
            a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            if ((a & 0x80u) != 0u)
                a = texa.ta1;
            else
                a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        default:
            break;
        }

        return (texel & 0x00FFFFFFu) | (static_cast<uint32_t>(a) << 24);
    }

    uint32_t addrPSMCT16Family(uint32_t basePtr, uint32_t width, uint8_t psm, uint32_t x, uint32_t y)
    {
        switch (psm)
        {
        case GS_PSM_CT16:
            return GSPSMCT16::addrPSMCT16(basePtr, width, x, y);
        case GS_PSM_CT16S:
            return GSPSMCT16::addrPSMCT16S(basePtr, width, x, y);
        case GS_PSM_Z16:
            return GSPSMCT16::addrPSMZ16(basePtr, width, x, y);
        case GS_PSM_Z16S:
            return GSPSMCT16::addrPSMZ16S(basePtr, width, x, y);
        default:
            return 0u;
        }
    }

    std::atomic<uint32_t> s_debugPrimitiveCount{0};
    std::atomic<uint32_t> s_debugPixelCount{0};
    std::atomic<uint32_t> s_debugContext1PrimitiveCount{0};
    std::atomic<uint32_t> s_debugFbp150PixelCount{0};

    int wrapTextureCoordinate(int coordinate,
                              int textureSize,
                              uint8_t mode,
                              uint16_t regionMin,
                              uint16_t regionMax)
    {
        switch (mode & 0x3u)
        {
        case 0: // REPEAT
            return static_cast<int>(static_cast<uint32_t>(coordinate) & static_cast<uint32_t>(textureSize - 1));
        case 1: // CLAMP
            return clampInt(coordinate, 0, textureSize - 1);
        case 2: // REGION_CLAMP
            return std::min(std::max(coordinate, static_cast<int>(regionMin)), static_cast<int>(regionMax));
        case 3: // REGION_REPEAT
            return static_cast<int>((static_cast<uint32_t>(coordinate) & static_cast<uint32_t>(regionMin)) | static_cast<uint32_t>(regionMax));
        default:
            return coordinate;
        }
    }

    bool passesAlphaTest(uint64_t testReg, uint8_t alpha)
    {
        if ((testReg & 0x1u) == 0u)
            return true;

        const uint8_t atst = static_cast<uint8_t>((testReg >> 1) & 0x7u);
        const uint8_t aref = static_cast<uint8_t>((testReg >> 4) & 0xFFu);

        switch (atst)
        {
        case 0:
            return false;
        case 1:
            return true;
        case 2:
            return alpha < aref;
        case 3:
            return alpha <= aref;
        case 4:
            return alpha == aref;
        case 5:
            return alpha >= aref;
        case 6:
            return alpha > aref;
        case 7:
            return alpha != aref;
        default:
            return true;
        }
    }

    struct PixelWriteMask
    {
        bool writeRgb = true;
        bool writeAlpha = true;
        bool writeDepth = true;

        bool writesFramebuffer() const
        {
            return writeRgb || writeAlpha;
        }

        bool writesAnything() const
        {
            return writesFramebuffer() || writeDepth;
        }
    };

    PixelWriteMask classifyAlphaTest(uint64_t testReg, uint8_t alpha, uint8_t framePsm)
    {
        const bool pass = passesAlphaTest(testReg, alpha);
        if (pass)
            return {};

        // TEST.AFAIL controls what happens when the alpha comparison fails.
        switch (static_cast<uint8_t>((testReg >> 12) & 0x3u))
        {
        case 1: // FB_ONLY
            return {true, true, false};
        case 2: // ZB_ONLY
            return {false, false, true};
        case 3: // RGB_ONLY
            // RGB_ONLY is only distinct for RGBA32. The GS treats it as
            // FB_ONLY for RGB24 and RGBA16 framebuffers.
            if (framePsm == GS_PSM_CT32)
                return {true, false, false};
            return {true, true, false};
        case 0: // KEEP
        default:
            return {false, false, false};
        }
    }

    bool passesDestinationAlphaTest(uint64_t testReg, uint8_t framePsm, uint32_t rawFramebufferPixel)
    {
        const bool date = ((testReg >> 14) & 0x1u) != 0u;
        if (!date)
            return true;

        const bool datm = ((testReg >> 15) & 0x1u) != 0u;
        switch (framePsm)
        {
        case GS_PSM_CT32:
            return (((rawFramebufferPixel >> 31) & 0x1u) != 0u) == datm;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            return (((rawFramebufferPixel >> 15) & 0x1u) != 0u) == datm;
        case GS_PSM_CT24:
            // RGB24 has no destination alpha, so DATE always passes.
            return true;
        default:
            return true;
        }
    }

    struct TextureCombineResult
    {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
    };

    TextureCombineResult combineTexture(const GSTex0Reg &tex,
                                        uint8_t vr,
                                        uint8_t vg,
                                        uint8_t vb,
                                        uint8_t va,
                                        uint8_t tr,
                                        uint8_t tg,
                                        uint8_t tb,
                                        uint8_t ta)
    {
        const bool textureHasAlpha = tex.tcc != 0u;
        TextureCombineResult out{tr, tg, tb, textureHasAlpha ? ta : va};

        switch (tex.tfx)
        {
        case 0: // MODULATE
            out.r = clampU8((tr * vr) >> 7);
            out.g = clampU8((tg * vg) >> 7);
            out.b = clampU8((tb * vb) >> 7);
            out.a = textureHasAlpha ? clampU8((ta * va) >> 7) : va;
            break;
        case 1: // DECAL
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : va;
            break;
        case 2: // HIGHLIGHT
            out.r = clampU8(((tr * vr) >> 7) + va);
            out.g = clampU8(((tg * vg) >> 7) + va);
            out.b = clampU8(((tb * vb) >> 7) + va);
            out.a = textureHasAlpha ? clampU8(ta + va) : va;
            break;
        case 3: // HIGHLIGHT2
            out.r = clampU8(((tr * vr) >> 7) + va);
            out.g = clampU8(((tg * vg) >> 7) + va);
            out.b = clampU8(((tb * vb) >> 7) + va);
            out.a = textureHasAlpha ? ta : va;
            break;
        default:
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : va;
            break;
        }

        return out;
    }

    uint32_t swizzleClutIndexCSM1(uint32_t index)
    {
        // CSM1 swaps address bits 3 and 4. Preserve the remaining bits:
        // 16-bit CLUTs expose a ninth address bit through CSA[4].
        return (index & ~0x18u) | ((index & 0x08u) << 1u) | ((index & 0x10u) >> 1u);
    }

    bool isFourBitIndexedPsm(uint8_t psm)
    {
        return psm == GS_PSM_T4 || psm == GS_PSM_T4HL || psm == GS_PSM_T4HH;
    }

    bool isEightBitIndexedPsm(uint8_t psm)
    {
        return psm == GS_PSM_T8 || psm == GS_PSM_T8H;
    }

    // Bilinear blend with 8-bit fixed-point weights (0..256): integer only, since the CRT
    // rounding calls it replaces dominated movie playback (four per channel per pixel).
    uint8_t lerpChannel(uint8_t c00, uint8_t c10, uint8_t c01, uint8_t c11, int fx, int fy)
    {
        const int top = (static_cast<int>(c00) << 8) + (static_cast<int>(c10) - static_cast<int>(c00)) * fx;
        const int bottom = (static_cast<int>(c01) << 8) + (static_cast<int>(c11) - static_cast<int>(c01)) * fx;
        return clampU8(((top << 8) + (bottom - top) * fy + 0x8000) >> 16);
    }

    int floorToInt(float value)
    {
        const int truncated = static_cast<int>(value);
        return value < static_cast<float>(truncated) ? truncated - 1 : truncated;
    }
}

namespace
{
    static constexpr uint32_t kDefaultDisplayWidth = 640u;
    static constexpr uint32_t kDefaultDisplayHeight = 448u;
    static constexpr uint32_t kHostFrameWidth = 640u;
    static constexpr uint32_t kHostFrameHeight = 512u;

    uint16_t encodeFramePixelPSMCT16(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        return static_cast<uint16_t>(((r >> 3) & 0x1Fu) |
                                     (((g >> 3) & 0x1Fu) << 5) |
                                     (((b >> 3) & 0x1Fu) << 10) |
                                     ((a >= 0x40u) ? 0x8000u : 0u));
    }

    void decodeDisplaySize(uint64_t display64, uint32_t &outWidth, uint32_t &outHeight)
    {
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display64 >> 23) & 0x0Fu);

        outWidth = (dw + 1u) / (magh + 1u);
        outHeight = dh + 1u;
        if (outWidth < 64u || outHeight < 64u)
        {
            outWidth = kDefaultDisplayWidth;
            outHeight = kDefaultDisplayHeight;
        }
        outWidth = std::min<uint32_t>(outWidth, kHostFrameWidth);
        outHeight = std::min<uint32_t>(outHeight, kHostFrameHeight);
    }

    GSFrameReg decodeDisplayFrame(uint64_t dispfb64)
    {
        GSFrameReg frame{};
        frame.fbp = static_cast<uint32_t>(dispfb64 & 0x1FFu);
        frame.fbw = static_cast<uint32_t>((dispfb64 >> 9) & 0x3Fu);
        frame.psm = static_cast<uint8_t>((dispfb64 >> 15) & 0x1Fu);
        return frame;
    }

    struct GSDisplayReadOrigin
    {
        uint32_t x = 0u;
        uint32_t y = 0u;
    };

    GSDisplayReadOrigin decodeDisplayReadOrigin(uint64_t dispfb64)
    {
        return {
            static_cast<uint32_t>((dispfb64 >> 32) & 0x7FFu),
            static_cast<uint32_t>((dispfb64 >> 43) & 0x7FFu)};
    }

    bool hasDisplaySetup(uint64_t display64, const GSFrameReg &frame)
    {
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display64 >> 23) & 0x0Fu);
        return frame.fbw != 0u || dw != 0u || dh != 0u || magh != 0u;
    }

    struct GSPmodeState
    {
        bool enableCrt1 = false;
        bool enableCrt2 = false;
        bool mmod = false;
        bool amod = false;
        bool slbg = false;
        uint8_t alp = 0u;
    };

    GSPmodeState decodePmode(uint64_t pmode64)
    {
        return {
            (pmode64 & 0x1ull) != 0ull,
            (pmode64 & 0x2ull) != 0ull,
            ((pmode64 >> 5) & 0x1ull) != 0ull,
            ((pmode64 >> 6) & 0x1ull) != 0ull,
            ((pmode64 >> 7) & 0x1ull) != 0ull,
            static_cast<uint8_t>((pmode64 >> 8) & 0xFFu)};
    }

    struct GSSmode2State
    {
        bool interlaced = false;
        bool frameMode = true;
    };

    GSSmode2State decodeSMode2(uint64_t smode2)
    {
        return {(smode2 & 0x1ull) != 0ull, ((smode2 >> 1) & 0x1ull) != 0ull};
    }

    void applyFieldPresentation(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height, bool oddField)
    {
        if (pixels.empty() || width == 0u || height < 2u)
            return;
        const std::vector<uint8_t> source = pixels;
        for (uint32_t y = 0; y < height; ++y)
        {
            uint32_t sourceY = ((y >> 1u) << 1u) + (oddField ? 1u : 0u);
            if (sourceY >= height)
                sourceY = height - 1u;
            std::memcpy(pixels.data() + y * kHostFrameWidth * 4u,
                        source.data() + sourceY * kHostFrameWidth * 4u,
                        width * 4u);
        }
    }

    // Interlaced FRAME mode (SMODE2 INT=1, FFMD=1) shows the same framebuffer lines in both fields,
    // so the buffer holds only (DH+1)/2 lines; scale them back up to the display height.
    void applyFrameModeLineDouble(std::vector<uint8_t> &pixels, uint32_t width, uint32_t sourceHeight, uint32_t targetHeight)
    {
        if (pixels.empty() || width == 0u || sourceHeight == 0u || targetHeight <= sourceHeight)
            return;
        for (uint32_t y = targetHeight; y-- > 0u;)
        {
            const uint32_t sourceY = std::min<uint32_t>((y * sourceHeight) / targetHeight, sourceHeight - 1u);
            std::memmove(pixels.data() + y * kHostFrameWidth * 4u,
                         pixels.data() + sourceY * kHostFrameWidth * 4u,
                         width * 4u);
        }
    }

    // Games in interlaced FRAME mode often alternate a sub-line offset every frame (one field draws
    // half a line lower) so a TV interleaves the fields; on a progressive display the picture then
    // shimmers up and down. While consecutive frames differ only by such a small amount, present the
    // average of the newest frame and the previous distinct one, which is identical every frame. Real
    // motion (larger differences) and long-static images are presented unblended.
    void applyFrameModeStabilize(std::vector<uint8_t> &pixels, uint32_t width, uint32_t sourceHeight, uint32_t targetHeight)
    {
        if (pixels.empty() || width == 0u || sourceHeight < 4u || targetHeight <= sourceHeight)
            return;

        constexpr uint32_t kStaticPresentsBeforeUnblend = 8u;
        constexpr uint32_t kMaxMeanDifference = 6u; // per sampled pixel, sum of RGB, after level removal

        const size_t rowBytes = kHostFrameWidth * 4u;
        const size_t usedBytes = rowBytes * sourceHeight;
        std::vector<uint8_t> other;
        bool recentlyChanged = false;
        {
            static std::mutex historyMutex;
            static std::vector<uint8_t> latest, previous;
            static uint32_t historyWidth = 0u, historyHeight = 0u;
            static uint32_t presentsSinceChange = 0u;
            std::lock_guard<std::mutex> lock(historyMutex);
            if (historyWidth != width || historyHeight != sourceHeight)
            {
                latest.clear();
                previous.clear();
                historyWidth = width;
                historyHeight = sourceHeight;
            }
            if (latest.size() != usedBytes || std::memcmp(latest.data(), pixels.data(), usedBytes) != 0)
            {
                previous.swap(latest);
                latest.assign(pixels.begin(), pixels.begin() + static_cast<std::ptrdiff_t>(usedBytes));
                presentsSinceChange = 0u;
            }
            else if (presentsSinceChange < kStaticPresentsBeforeUnblend)
            {
                ++presentsSinceChange;
            }
            recentlyChanged = presentsSinceChange < kStaticPresentsBeforeUnblend;
            if (recentlyChanged && previous.size() == usedBytes)
                other = previous;
        }

        if (!other.empty())
        {
            auto level = [&](const uint8_t *frame)
            {
                uint64_t sum = 0u, count = 0u;
                for (uint32_t y = 0; y < sourceHeight; y += 2u)
                    for (uint32_t x = 0; x < width; x += 4u, ++count)
                    {
                        const uint8_t *p = frame + y * rowBytes + x * 4u;
                        sum += p[0] + p[1] + p[2];
                    }
                return count ? static_cast<int64_t>(sum / count) : 0;
            };
            const int64_t levelCurrent = level(pixels.data());
            const int64_t levelOther = level(other.data());
            // Compare the change since the previous frame with what shifting this image by one line
            // costs: a field offset (up to a line) stays within that, real motion usually does not.
            uint64_t difference = 0u, oneLineShift = 0u, samples = 0u;
            for (uint32_t y = 0; y + 1u < sourceHeight; y += 2u)
                for (uint32_t x = 0; x < width; x += 4u, ++samples)
                {
                    const uint8_t *a = pixels.data() + y * rowBytes + x * 4u;
                    const uint8_t *below = a + rowBytes;
                    const uint8_t *b = other.data() + y * rowBytes + x * 4u;
                    difference += static_cast<uint64_t>(std::llabs((int64_t(a[0]) + a[1] + a[2] - levelCurrent) -
                                                                   (int64_t(b[0]) + b[1] + b[2] - levelOther)));
                    oneLineShift += static_cast<uint64_t>(std::llabs((int64_t(a[0]) + a[1] + a[2]) -
                                                                     (int64_t(below[0]) + below[1] + below[2])));
                }
            const uint64_t allowed = std::max<uint64_t>(samples * kMaxMeanDifference, oneLineShift + oneLineShift / 2u);
            if (samples != 0u && difference <= allowed)
            {
                for (uint32_t y = 0; y < sourceHeight; ++y)
                {
                    uint8_t *dst = pixels.data() + y * rowBytes;
                    const uint8_t *src = other.data() + y * rowBytes;
                    for (uint32_t i = 0; i < width * 4u; ++i)
                        dst[i] = static_cast<uint8_t>((uint32_t(dst[i]) + src[i] + 1u) >> 1u);
                }
            }
        }

        applyFrameModeLineDouble(pixels, width, sourceHeight, targetHeight);
    }

    void normalizePresentationAlpha(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        for (uint32_t y = 0; y < height; ++y)
        {
            uint8_t *row = pixels.data() + y * kHostFrameWidth * 4u;
            for (uint32_t x = 0; x < width; ++x)
                row[x * 4u + 3u] = 255u;
        }
    }

    uint8_t blendPresentationChannel(uint8_t src, uint8_t dst, uint32_t factor)
    {
        const int delta = static_cast<int>(src) - static_cast<int>(dst);
        return GSInternal::clampU8(static_cast<int>(dst) + ((delta * static_cast<int>(factor)) / 255));
    }

    uint32_t countNonBlackPixels(const std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        uint32_t count = 0u;
        for (uint32_t y = 0; y < height; ++y)
        {
            const uint8_t *row = pixels.data() + y * kHostFrameWidth * 4u;
            for (uint32_t x = 0; x < width; ++x)
            {
                if (row[x * 4u] != 0u || row[x * 4u + 1u] != 0u || row[x * 4u + 2u] != 0u)
                    ++count;
            }
        }
        return count;
    }
}

namespace
{
    // Worker spin before sleeping, about 50 us: primitives of a frame arrive back to back.
    constexpr uint32_t kWorkerSpinIterations = 2000u;
    constexpr uint64_t kQueueSize = 4096u; // power of two
    constexpr uint32_t kVramBlocks = 16384u; // 4 MiB in 256-byte blocks
    constexpr size_t kMaxPendingWrites = 32u;
    constexpr uint32_t kClutSnapshots = 256u;
    constexpr int kDecodedTileShift = 5; // 32x32 texel tiles
    constexpr size_t kDecodedTexturesPerWorker = 16u;
    constexpr int kDecodedMaxTexels = 1024 * 1024;
    // Smaller primitives (text, particles) sample the texture directly: decoding a whole
    // tile would cost more than the pixels it serves.
    constexpr int64_t kDecodedMinPixels = 1024;

    inline void cpuRelax()
    {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
        _mm_pause();
#else
        std::this_thread::yield();
#endif
    }

    // Page size in pixels of a pixel storage format (the H formats live in CT32 pages).
    void pageGeometry(uint32_t psm, uint32_t &width, uint32_t &height)
    {
        switch (psm)
        {
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
            width = 64u;
            height = 64u;
            break;
        case GS_PSM_T8:
            width = 128u;
            height = 64u;
            break;
        case GS_PSM_T4:
            width = 128u;
            height = 128u;
            break;
        default:
            width = 64u;
            height = 32u;
            break;
        }
    }

    // VRAM blocks of the whole page rows that hold rows [firstRow, lastRow] of a buffer.
    std::pair<uint32_t, uint32_t> rowBlocks(uint32_t baseBlock, uint32_t width64, uint32_t psm, int firstRow, int lastRow)
    {
        uint32_t pageWidth = 0u;
        uint32_t pageHeight = 0u;
        pageGeometry(psm, pageWidth, pageHeight);
        const uint32_t pagesPerRow = std::max(1u, (std::max(width64, 1u) * 64u + pageWidth - 1u) / pageWidth);
        const uint32_t first = static_cast<uint32_t>(std::max(firstRow, 0)) / pageHeight;
        const uint32_t last = static_cast<uint32_t>(std::max(lastRow, std::max(firstRow, 0))) / pageHeight;
        const uint64_t count = static_cast<uint64_t>(last - first + 1u) * pagesPerRow * 32u;
        return {(baseBlock + first * pagesPerRow * 32u) % kVramBlocks,
                static_cast<uint32_t>(std::min<uint64_t>(kVramBlocks, count))};
    }

    uint32_t primitiveVertexCount(uint8_t type)
    {
        switch (type)
        {
        case GS_PRIM_POINT:
            return 1u;
        case GS_PRIM_LINE:
        case GS_PRIM_LINESTRIP:
        case GS_PRIM_SPRITE:
            return 2u;
        default:
            return 3u;
        }
    }

    // Pixels of the primitive's bounding box within the scissor.
    int64_t primitiveArea(const GSPrimitiveBatch &batch)
    {
        const auto &ctx = batch.state.context;
        const float ofx = static_cast<float>(ctx.xyoffset.ofx >> 4);
        const float ofy = static_cast<float>(ctx.xyoffset.ofy >> 4);
        const uint32_t count = std::min<uint32_t>(primitiveVertexCount(batch.state.prim.type), batch.vertexCount);
        float minX = batch.vertices[0].x, maxX = minX, minY = batch.vertices[0].y, maxY = minY;
        for (uint32_t i = 1; i < count; ++i)
        {
            minX = std::min(minX, batch.vertices[i].x);
            maxX = std::max(maxX, batch.vertices[i].x);
            minY = std::min(minY, batch.vertices[i].y);
            maxY = std::max(maxY, batch.vertices[i].y);
        }
        const int x0 = clampInt(static_cast<int>(minX - ofx), ctx.scissor.x0, ctx.scissor.x1);
        const int x1 = clampInt(static_cast<int>(maxX - ofx), ctx.scissor.x0, ctx.scissor.x1);
        const int y0 = clampInt(static_cast<int>(minY - ofy), ctx.scissor.y0, ctx.scissor.y1);
        const int y1 = clampInt(static_cast<int>(maxY - ofy), ctx.scissor.y0, ctx.scissor.y1);
        return static_cast<int64_t>(x1 - x0 + 1) * static_cast<int64_t>(y1 - y0 + 1);
    }

    // Framebuffer rows a primitive can touch (its bounding box within the scissor).
    void primitiveRows(const GSPrimitiveBatch &batch, int &firstRow, int &lastRow)
    {
        const auto &ctx = batch.state.context;
        const float ofy = static_cast<float>(ctx.xyoffset.ofy >> 4);
        const uint32_t count = std::min<uint32_t>(primitiveVertexCount(batch.state.prim.type), batch.vertexCount);
        float minY = batch.vertices[0].y;
        float maxY = minY;
        for (uint32_t i = 1; i < count; ++i)
        {
            minY = std::min(minY, batch.vertices[i].y);
            maxY = std::max(maxY, batch.vertices[i].y);
        }
        firstRow = clampInt(static_cast<int>(minY - ofy) - 1, ctx.scissor.y0, ctx.scissor.y1);
        lastRow = clampInt(static_cast<int>(maxY - ofy) + 1, ctx.scissor.y0, ctx.scissor.y1);
    }

    // Texture rows a primitive can sample. Perspective-correct T/Q stays within the vertices'
    // range while Q keeps one sign; wrapping or mixed signs mean the whole texture.
    // Texel row DrawSprite samples for pixel row `y` of an fst sprite (its fixed-point UV and,
    // when filtering, the upper bilinear tap).
    int spriteTexelRow(const GSPrimitiveBatch &batch, int y, int unclippedY0, int spanY, bool linear)
    {
        const float v0f = static_cast<float>(batch.vertices[0].v) / 16.0f;
        const float v1f = static_cast<float>(batch.vertices[1].v) / 16.0f;
        const float ty = (static_cast<float>(y - unclippedY0) + 0.5f) / static_cast<float>(std::max(spanY, 1));
        const float texVf = v0f + (v1f - v0f) * ty;
        const int fixedV = clampInt(static_cast<int>((texVf * 16.0f) + 0.5f), 0, 0xFFFF);
        const float sampleV = static_cast<float>(fixedV) / 16.0f;
        return linear ? floorToInt(sampleV - 0.5f) : static_cast<int>(sampleV);
    }

    // The highest row a bilinear sample can weigh: none below its own when it lies on a texel center.
    int spriteTexelLastRow(const GSPrimitiveBatch &batch, int y, int unclippedY0, int spanY)
    {
        const float v0f = static_cast<float>(batch.vertices[0].v) / 16.0f;
        const float v1f = static_cast<float>(batch.vertices[1].v) / 16.0f;
        const float ty = (static_cast<float>(y - unclippedY0) + 0.5f) / static_cast<float>(std::max(spanY, 1));
        const float texVf = v0f + (v1f - v0f) * ty;
        const int fixedV = clampInt(static_cast<int>((texVf * 16.0f) + 0.5f), 0, 0xFFFF);
        return static_cast<int>(std::ceil(static_cast<float>(fixedV) / 16.0f - 0.5f));
    }

    void textureRows(const GSPrimitiveBatch &batch, int &firstRow, int &lastRow)
    {
        const GSDrawState &state = batch.state;
        const int height = std::max<int>(state.textureHeight, 1);
        firstRow = 0;
        lastRow = height - 1;
        if (state.prim.type == GS_PRIM_SPRITE && state.prim.fst && batch.vertexCount >= 2u)
        {
            // Exact rows instead of the +-2 margin: a full-screen copy starting at row 0 would
            // otherwise wrap (REPEAT) onto the whole texture height.
            const float ofy = static_cast<float>(state.context.xyoffset.ofy) / 16.0f;
            const float fy0 = std::min(batch.vertices[0].y, batch.vertices[1].y) - ofy;
            const float fy1 = std::max(batch.vertices[0].y, batch.vertices[1].y) - ofy;
            const int unclippedY0 = static_cast<int>(std::ceil(fy0));
            const int unclippedY1 = std::max(unclippedY0, static_cast<int>(std::ceil(fy1)) - 1);
            const int spanY = unclippedY1 - unclippedY0 + 1;
            const bool linear = state.linearFilter;
            const int a = spriteTexelRow(batch, unclippedY0, unclippedY0, spanY, linear);
            const int b = spriteTexelRow(batch, unclippedY1, unclippedY0, spanY, linear);
            const int low = std::min(a, b);
            const int high = linear ? std::max(spriteTexelLastRow(batch, unclippedY0, unclippedY0, spanY),
                                               spriteTexelLastRow(batch, unclippedY1, unclippedY0, spanY))
                                    : std::max(a, b);
            const uint8_t wrapV = static_cast<uint8_t>((state.context.clamp >> 2) & 0x3u);
            // CLAMP and REGION_CLAMP clamp out-of-range rows; REPEAT and REGION_REPEAT may wrap them.
            int minRow = 0;
            int maxRow = height - 1;
            if (wrapV == 2u)
            {
                minRow = clampInt(static_cast<int>((state.context.clamp >> 24) & 0x3FFu), 0, height - 1);
                maxRow = clampInt(static_cast<int>((state.context.clamp >> 34) & 0x3FFu), minRow, height - 1);
            }
            if (wrapV == 1u || wrapV == 2u || (wrapV == 0u && low >= 0 && high < height))
            {
                firstRow = clampInt(low, minRow, maxRow);
                lastRow = clampInt(high, minRow, maxRow);
            }
            return;
        }
        const uint32_t count = std::min<uint32_t>(primitiveVertexCount(state.prim.type), batch.vertexCount);
        float minV = 0.0f;
        float maxV = 0.0f;
        bool positiveQ = false;
        bool negativeQ = false;
        for (uint32_t i = 0; i < count; ++i)
        {
            const GSVertex &v = batch.vertices[i];
            float row = 0.0f;
            if (state.prim.fst)
            {
                row = static_cast<float>(v.v) / 16.0f;
            }
            else
            {
                positiveQ |= v.q > 0.0f;
                negativeQ |= v.q <= 0.0f;
                row = (v.q != 0.0f ? v.t / v.q : 0.0f) * static_cast<float>(height);
            }
            minV = i == 0 ? row : std::min(minV, row);
            maxV = i == 0 ? row : std::max(maxV, row);
        }
        if (!state.prim.fst && (negativeQ || !positiveQ))
            return;
        const int low = static_cast<int>(minV) - 2; // bilinear taps and rounding
        const int high = static_cast<int>(maxV) + 2;
        const uint8_t wrapV = static_cast<uint8_t>((state.context.clamp >> 2) & 0x3u);
        // REGION_CLAMP/REGION_REPEAT remap rows to the region; REPEAT wraps out-of-range rows.
        if (wrapV >= 2u || (wrapV == 0u && (low < 0 || high >= height)))
            return;
        firstRow = std::max(low, 0);
        lastRow = std::min(high, height - 1);
    }

    bool blocksOverlap(std::pair<uint32_t, uint32_t> a, std::pair<uint32_t, uint32_t> b)
    {
        // VRAM addresses wrap at 4 MiB.
        for (const int64_t shift : {-static_cast<int64_t>(kVramBlocks), int64_t{0}, static_cast<int64_t>(kVramBlocks)})
        {
            const int64_t bStart = static_cast<int64_t>(b.first) + shift;
            if (static_cast<int64_t>(a.first) < bStart + b.second && bStart < static_cast<int64_t>(a.first) + a.second)
                return true;
        }
        return false;
    }

    bool overlapsAny(std::pair<uint32_t, uint32_t> range, const std::vector<std::pair<uint32_t, uint32_t>> &ranges)
    {
        for (const auto &other : ranges)
        {
            if (blocksOverlap(range, other))
                return true;
        }
        return false;
    }

    // VRAM blocks a primitive can write (whole page rows of its frame and Z buffers) and sample.
    struct PrimitiveBlocks
    {
        std::pair<uint32_t, uint32_t> frame;
        std::pair<uint32_t, uint32_t> z;
        std::pair<uint32_t, uint32_t> texture;
        bool writesZ = false;
        bool samples = false;
    };

    PrimitiveBlocks primitiveBlocks(const GSPrimitiveBatch &batch)
    {
        const auto &ctx = batch.state.context;
        PrimitiveBlocks blocks;
        int firstRow = 0;
        int lastRow = 0;
        primitiveRows(batch, firstRow, lastRow);
        blocks.frame = rowBlocks(GSInternal::framePageBaseToBlock(ctx.frame.fbp), ctx.frame.fbw, ctx.frame.psm, firstRow, lastRow);
        blocks.writesZ = !ctx.zbuf.zmask;
        if (blocks.writesZ)
            blocks.z = rowBlocks(GSInternal::framePageBaseToBlock(ctx.zbuf.zbp), ctx.frame.fbw, ctx.zbuf.psm, firstRow, lastRow);
        blocks.samples = batch.state.prim.tme;
        if (blocks.samples)
        {
            textureRows(batch, firstRow, lastRow);
            blocks.texture = rowBlocks(ctx.tex0.tbp0, ctx.tex0.tbw, ctx.tex0.psm, firstRow, lastRow);
        }
        return blocks;
    }

    // Samples what it draws: the result depends on the order its pixels are written in.
    bool readsOwnWrites(const PrimitiveBlocks &blocks)
    {
        return blocks.samples && (blocksOverlap(blocks.texture, blocks.frame) ||
                                  (blocks.writesZ && blocksOverlap(blocks.texture, blocks.z)));
    }
}

struct GSCpuBackend::PixelPipeline
{
    int scissorX0 = 0;
    int scissorY0 = 0;
    int scissorX1 = 0;
    int scissorY1 = 0;
    bool fge = false;
    uint8_t fogR = 0;
    uint8_t fogG = 0;
    uint8_t fogB = 0;
    uint32_t fbp = 0;
    uint32_t fbw = 1;
    uint32_t zbp = 0;
    uint8_t fpsm = 0;
    uint8_t zpsm = 0;
    uint64_t test = 0;
    uint32_t ztst = 0;
    bool alphaTest = false;
    bool dateNeedsRead = false;
    bool passReadsFramebuffer = false; // framebuffer read when the alpha test passes
    bool abe = false;
    bool pabe = false;
    uint8_t asel = 0;
    uint8_t bsel = 0;
    uint8_t csel = 0;
    uint8_t dsel = 0;
    uint8_t fix = 0;
    bool fba = false; // FBA on a frame format that stores alpha
    uint32_t fbmsk = 0;
    bool zmask = false;
    bool frame16 = false;
    bool frame24 = false;
    ReadVramFunc readFrame = nullptr;
    ReadVramFunc readZ = nullptr;
    WriteVramFunc writeFrame = nullptr;
    WriteVramFunc writeZ = nullptr;
};

struct GSCpuBackend::TextureSampler
{
    const GSDrawState *state = nullptr;
    int texW = 1;
    int texH = 1;
    uint8_t wrapU = 0;
    uint8_t wrapV = 0;
    uint16_t minU = 0;
    uint16_t maxU = 0;
    uint16_t minV = 0;
    uint16_t maxV = 0;
    bool fst = false;
    bool linear = false;
    const std::array<uint16_t, 512> *clut = nullptr;
    uint64_t clutSerial = 0;                // identifies the CLUT contents
    DecodedTexture *decoded = nullptr;      // worker's decoded copy, when async
};

struct GSCpuBackend::RasterContext
{
    const PixelPipeline *pixel;
    const TextureSampler *texture;
    GSMem::TexturePageCache *cache;
    uint32_t stripe;
    uint32_t stripeCount;
    // Page geometry of the frame buffer being drawn: a worker owns the VRAM pages whose
    // number modulo stripeCount is its stripe, so each word (whatever buffer geometry or
    // format writes it) is always written by the same worker, in queue order.
    uint32_t basePage = 0;
    uint32_t pagesPerRow = 1;
    uint32_t pageWidth = 64;
    uint32_t pageHeight = 32;

    bool owns(int x, int y) const
    {
        if (stripeCount <= 1u)
            return true;
        const uint32_t page = basePage + (static_cast<uint32_t>(y) / pageHeight) * pagesPerRow + static_cast<uint32_t>(x) / pageWidth;
        return (page % stripeCount) == stripe;
    }

    // Narrows [start, limit] on row y to its first run of owned pixels, in end.
    bool ownedRun(int y, int &start, int &end, int limit) const
    {
        if (stripeCount <= 1u)
        {
            end = limit;
            return start <= limit;
        }
        const uint32_t rowPage = basePage + (static_cast<uint32_t>(y) / pageHeight) * pagesPerRow;
        while (start <= limit)
        {
            const uint32_t column = static_cast<uint32_t>(start) / pageWidth;
            const int columnEnd = std::min(limit, static_cast<int>((column + 1u) * pageWidth) - 1);
            if (((rowPage + column) % stripeCount) == stripe)
            {
                end = columnEnd;
                return true;
            }
            start = columnEnd + 1;
        }
        return false;
    }
};

struct GSCpuBackend::QueuedPrimitive
{
    GSPrimitiveBatch batch;
    PixelPipeline pixel;
    TextureSampler texture; // its state pointer is rebound to batch.state by each worker
    bool textureFlush = false; // TEXFLUSH marker instead of a primitive
    bool barrier = false;      // samples what earlier queued primitives draw: wait for them
    int64_t area = 0;          // bounding box pixels
    bool upload = false;       // host-to-local image data instead of a primitive
    GSTransferCommand transfer{};
    uint32_t uploadFirstPixel = 0;
    uint32_t uploadTotalPixels = 0;
    std::vector<uint8_t> uploadData;
    // VRAM blocks it writes (frame or upload target, depth) and samples.
    std::pair<uint32_t, uint32_t> writes[2]{};
    uint8_t writeCount = 0;
    std::pair<uint32_t, uint32_t> reads{};
    bool samples = false;
};

// Texels converted to RGBA (TEXA and CLUT applied) in a linear array, filled by 32x32 tiles on
// first use: bilinear filtering then reads an array instead of four swizzled, palettized
// fetches per pixel. Like the texture page cache, its contents stay until TEXFLUSH.
struct GSCpuBackend::DecodedTexture
{
    uint32_t tbp0 = 0;
    uint32_t tbw = 0;
    uint8_t psm = 0;
    uint8_t cpsm = 0;
    uint8_t csm = 0;
    uint8_t csa = 0;
    int width = 0;
    int height = 0;
    uint64_t clutSerial = 0;
    GSTexaReg texa{};
    uint64_t lastUse = 0;
    uint32_t tilesPerRow = 0;
    uint64_t writeClock = 0; // NoteDirectWrite clock its tiles are newer than (immediate drawing)
    bool valid = false;
    std::unique_ptr<uint32_t[]> texels; // not cleared: tiles are written before use
    size_t texelCapacity = 0;
    std::vector<uint8_t> tileReady;
};

struct GSCpuBackend::DecodedTextureCache
{
    std::vector<DecodedTexture> textures;
    uint64_t clock = 0;

    // Keeps the buffers: TEXFLUSH is frequent and reallocating them dominated.
    void Invalidate()
    {
        for (DecodedTexture &texture : textures)
            texture.valid = false;
    }
};

GSCpuBackend::PixelPipeline GSCpuBackend::MakePixelPipeline(const GSDrawState &state) const
{
    const auto &ctx = state.context;
    PixelPipeline p;
    p.scissorX0 = ctx.scissor.x0;
    p.scissorY0 = ctx.scissor.y0;
    p.scissorX1 = ctx.scissor.x1;
    p.scissorY1 = ctx.scissor.y1;
    p.fge = state.prim.fge;
    p.fogR = state.fogR;
    p.fogG = state.fogG;
    p.fogB = state.fogB;
    p.fbp = GSInternal::framePageBaseToBlock(ctx.frame.fbp);
    p.fbw = std::max<u32>(ctx.frame.fbw, 1u);
    p.fpsm = static_cast<uint8_t>(ctx.frame.psm);
    p.zbp = GSInternal::framePageBaseToBlock(ctx.zbuf.zbp);
    p.zpsm = static_cast<uint8_t>(ctx.zbuf.psm);
    p.test = ctx.test;
    p.ztst = static_cast<uint32_t>((ctx.test >> 17) & 3u);
    p.alphaTest = (ctx.test & 0x1u) != 0u;
    p.dateNeedsRead = ((ctx.test >> 14) & 0x1u) != 0u &&
                      (p.fpsm == GS_PSM_CT32 || p.fpsm == GS_PSM_CT16 || p.fpsm == GS_PSM_CT16S);
    p.abe = state.prim.abe;
    p.pabe = state.pabe;
    p.asel = static_cast<uint8_t>(ctx.alpha & 3u);
    p.bsel = static_cast<uint8_t>((ctx.alpha >> 2) & 3u);
    p.csel = static_cast<uint8_t>((ctx.alpha >> 4) & 3u);
    p.dsel = static_cast<uint8_t>((ctx.alpha >> 6) & 3u);
    p.fix = static_cast<uint8_t>((ctx.alpha >> 32) & 0xFFu);
    p.fba = (ctx.fba & 0x1ull) != 0ull && ctx.frame.psm != GS_PSM_CT24;
    p.fbmsk = ctx.frame.fbmsk;
    p.zmask = ctx.zbuf.zmask != 0;
    p.passReadsFramebuffer = p.dateNeedsRead || p.fbmsk != 0u || p.abe;
    p.frame16 = bitsPerPixel(p.fpsm) == 16;
    p.frame24 = p.fpsm == GS_PSM_CT24;
    p.readFrame = m_readVramFuncs[p.fpsm & 0x3Fu];
    p.readZ = m_readVramFuncs[p.zpsm & 0x3Fu];
    p.writeFrame = m_writeVramFuncs[p.fpsm & 0x3Fu];
    p.writeZ = m_writeVramFuncs[p.zpsm & 0x3Fu];
    return p;
}

GSCpuBackend::TextureSampler GSCpuBackend::MakeTextureSampler(const GSDrawState &state) const
{
    const uint64_t clamp = state.context.clamp;
    TextureSampler s;
    s.state = &state;
    s.texW = state.textureWidth;
    s.texH = state.textureHeight;
    s.wrapU = static_cast<uint8_t>(clamp & 0x3u);
    s.wrapV = static_cast<uint8_t>((clamp >> 2) & 0x3u);
    s.minU = static_cast<uint16_t>((clamp >> 4) & 0x3FFu);
    s.maxU = static_cast<uint16_t>((clamp >> 14) & 0x3FFu);
    s.minV = static_cast<uint16_t>((clamp >> 24) & 0x3FFu);
    s.maxV = static_cast<uint16_t>((clamp >> 34) & 0x3FFu);
    s.fst = state.prim.fst;
    s.linear = state.linearFilter;
    s.clut = &m_clut;
    s.clutSerial = m_clutSerial;
    return s;
}

GSCpuBackend::GSCpuBackend()
{
    using namespace GSMem;
    static std::once_flag lookupTablesOnce;
    std::call_once(lookupTablesOnce, []()
                   { InitLookupTables(); });
    for (size_t i = 0; i < kPsmHandlerCount; ++i)
    {
        switch (i)
        {
        case GS_PSM_CT32:
            m_readVramFuncs[i] = ReadCT32;
            m_writeVramFuncs[i] = WriteCT32;
            break;
        case GS_PSM_CT24:
            m_readVramFuncs[i] = ReadCT24;
            m_writeVramFuncs[i] = WriteCT24;
            break;
        case GS_PSM_CT16:
            m_readVramFuncs[i] = ReadCT16;
            m_writeVramFuncs[i] = WriteCT16;
            break;
        case GS_PSM_CT16S:
            m_readVramFuncs[i] = ReadCT16S;
            m_writeVramFuncs[i] = WriteCT16S;
            break;
        case GS_PSM_T8:
            m_readVramFuncs[i] = ReadP8;
            m_writeVramFuncs[i] = WriteP8;
            break;
        case GS_PSM_T8H:
            m_readVramFuncs[i] = ReadP8H;
            m_writeVramFuncs[i] = WriteP8H;
            break;
        case GS_PSM_T4:
            m_readVramFuncs[i] = ReadP4;
            m_writeVramFuncs[i] = WriteP4;
            break;
        case GS_PSM_T4HH:
            m_readVramFuncs[i] = ReadP4HH;
            m_writeVramFuncs[i] = WriteP4HH;
            break;
        case GS_PSM_T4HL:
            m_readVramFuncs[i] = ReadP4HL;
            m_writeVramFuncs[i] = WriteP4HL;
            break;
        case GS_PSM_Z32:
            m_readVramFuncs[i] = ReadZ32;
            m_writeVramFuncs[i] = WriteZ32;
            break;
        case GS_PSM_Z24:
            m_readVramFuncs[i] = ReadZ24;
            m_writeVramFuncs[i] = WriteZ24;
            break;
        case GS_PSM_Z16:
            m_readVramFuncs[i] = ReadZ16;
            m_writeVramFuncs[i] = WriteZ16;
            break;
        case GS_PSM_Z16S:
            m_readVramFuncs[i] = ReadZ16S;
            m_writeVramFuncs[i] = WriteZ16S;
            break;
        default:
            m_readVramFuncs[i] = ReadNull;
            m_writeVramFuncs[i] = WriteNull;
            break;
        }
    }
    Reset();
    m_texturePageCaches.resize(1u);
    m_decodedCaches.resize(1u);
}

GSCpuBackend::~GSCpuBackend()
{
    if (m_workerCount != 0u)
        DrainUnlocked();
    {
        std::lock_guard<std::mutex> lock(m_workMutex);
        m_stopWorkers.store(true, std::memory_order_release);
    }
    m_workCv.notify_all();
    for (std::thread &worker : m_workers)
        worker.join();
}

void GSCpuBackend::Initialize(uint8_t *vram, uint32_t vramSize)
{
    if (vram && vramSize < GSMem::MEMORY_SIZE)
        throw std::invalid_argument("GS CPU backend requires at least 4 MiB of VRAM");

    std::lock_guard<std::mutex> lock(m_mutex);
    DrainUnlocked();
    m_vram = vram;
    m_vramSize = vramSize;
    ResetUnlocked();
}

void GSCpuBackend::Reset()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    DrainUnlocked();
    ResetUnlocked();
}

void GSCpuBackend::ResetUnlocked()
{
    m_clut.fill(0u);
    m_clutSnapshotStale = true;
    ++m_clutSerial;
    for (DecodedTextureCache &cache : m_decodedCaches)
        cache.Invalidate();
    m_clutCbp.fill(0u);
    for (GSMem::TexturePageCache &cache : m_texturePageCaches)
        cache.Invalidate();
    m_transfer = {};
    m_transfer.direction = 3u;
    m_transferState = {};
    m_transferState.direction = 3u;
    m_localToHostBuffer.clear();
    m_localToHostReadPos = 0u;
    m_pixelWritesSinceBarrier.clear();
    m_zWritesSinceBarrier.clear();
    m_barrierRequested = false;
    m_callerCacheStale = false;
}

// VU1 microprogram (MSCAL start pc) whose XGKICK is being drawn, for PS2X_GS_DUMP_AT; ~0u outside VU1.
std::atomic<uint32_t> g_gsDumpVu1ProgramPc{~0u};
// GIF path (1-3) of the packet being processed, for PS2X_GS_DUMP_AT; 0 when unknown.
std::atomic<uint32_t> g_gsDumpGifPath{0u};

// PS2X_GS_DUMP_AT=<sec>: once `sec` seconds have passed, the GS writes its next PS2X_GS_DUMP_COUNT (default 20000)
// primitives and the GIF tags that produced them to PS2X_GS_DUMP_OUT (default gs_prims.txt). Returns the open
// stream while the dump is in progress, else nullptr.
FILE *gsDumpStream()
{
    static const double dumpAt = []
    {
        const char *value = std::getenv("PS2X_GS_DUMP_AT");
        return value ? std::atof(value) : -1.0;
    }();
    if (dumpAt < 0.0)
        return nullptr;
    static const auto start = std::chrono::steady_clock::now();
    static FILE *out = nullptr;
    static bool finished = false;
    if (finished || std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < dumpAt)
        return nullptr;
    if (!out)
    {
        const char *path = std::getenv("PS2X_GS_DUMP_OUT");
        out = std::fopen(path ? path : "gs_prims.txt", "w");
        finished = out == nullptr;
    }
    return out;
}

namespace
{
    void dumpPrimitiveIfRequested(const GSPrimitiveBatch &batch)
    {
        FILE *out = gsDumpStream();
        if (!out)
            return;
        static uint32_t remaining = []
        {
            const char *value = std::getenv("PS2X_GS_DUMP_COUNT");
            return value ? static_cast<uint32_t>(std::atoi(value)) : 20000u;
        }();
        if (remaining == 0u)
            return;
        const GSDrawState &state = batch.state;
        const GSContext &ctx = state.context;
        std::fprintf(out, "vu1pc=%x type=%u tme=%u fst=%u abe=%u fge=%u ctxt=%u fbp=%u psm=%x zbp=%u test=%llx alpha=%llx "
                          "ofs=(%u,%u) sc=(%u,%u)-(%u,%u) tbp=%u tpsm=%x |",
                     g_gsDumpVu1ProgramPc.load(std::memory_order_relaxed), static_cast<uint32_t>(state.prim.type),
                     state.prim.tme, state.prim.fst, state.prim.abe,
                     state.prim.fge, state.prim.ctxt, ctx.frame.fbp, static_cast<uint32_t>(ctx.frame.psm),
                     ctx.zbuf.zbp, static_cast<unsigned long long>(ctx.test),
                     static_cast<unsigned long long>(ctx.alpha), ctx.xyoffset.ofx >> 4, ctx.xyoffset.ofy >> 4,
                     ctx.scissor.x0, ctx.scissor.y0, ctx.scissor.x1, ctx.scissor.y1, ctx.tex0.tbp0,
                     static_cast<uint32_t>(ctx.tex0.psm));
        for (uint32_t i = 0; i < batch.vertexCount && i < 3u; ++i)
        {
            const GSVertex &v = batch.vertices[i];
            std::fprintf(out, " (%.1f,%.1f,%.0f c=%02x%02x%02x%02x f=%u stq=%g,%g,%g uv=%u,%u)", v.x, v.y, v.z, v.r,
                         v.g, v.b, v.a, v.fog, v.s, v.t, v.q, v.u, v.v);
        }
        std::fputc('\n', out);
        if (--remaining == 0u)
            std::fflush(out);
    }
}

void GSCpuBackend::Submit(const GSPrimitiveBatch &batch)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || batch.vertexCount == 0u)
        return;
    if (s_asyncRasterization.load(std::memory_order_relaxed) && m_asyncAllowed)
        StartWorkersUnlocked();
    DumpUpdateUnlocked();
    dumpPrimitiveIfRequested(batch);
    DrawPrimitive(batch);
}

void GSCpuBackend::LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || (!isFourBitIndexedPsm(tex0.psm) && !isEightBitIndexedPsm(tex0.psm)))
        return;
    if (m_dumpRemaining != 0u)
        DumpLog("LOADCLUT cld=%u cbp=%u cpsm=0x%x csm=%u csa=%u psm=0x%x cbw=%u cou=%u cov=%u", unsigned(tex0.cld), unsigned(tex0.cbp),
                unsigned(tex0.cpsm), unsigned(tex0.csm), unsigned(tex0.csa), unsigned(tex0.psm), unsigned(texclut.cbw),
                unsigned(texclut.cou), unsigned(texclut.cov));

    switch (tex0.cld)
    {
    case 0u:
    case 6u:
    case 7u:
        return;
    case 1u:
        break;
    case 2u:
        m_clutCbp[0] = tex0.cbp;
        break;
    case 3u:
        m_clutCbp[1] = tex0.cbp;
        break;
    case 4u:
        if (m_clutCbp[0] == tex0.cbp)
            return;
        m_clutCbp[0] = tex0.cbp;
        break;
    case 5u:
        if (m_clutCbp[1] == tex0.cbp)
            return;
        m_clutCbp[1] = tex0.cbp;
        break;
    default:
        return;
    }

    if (m_workerCount != 0u)
    {
        // Queued primitives sample CLUT snapshots, so only the source rows matter.
        const auto source = tex0.csm == 0u ? rowBlocks(tex0.cbp, 1u, tex0.cpsm, 0, 15)
                                           : rowBlocks(tex0.cbp, texclut.cbw, tex0.cpsm, texclut.cov, texclut.cov);
        if (OverlapsInFlightUnlocked(source, false))
            DrainUnlocked();
        // Only CLUT loads use this cache while workers draw, so it would keep pages they have
        // written since far longer than the immediate rasterizer's does.
        if (m_callerCacheStale)
        {
            m_texturePageCaches[0].Invalidate();
            m_decodedCaches[0].Invalidate();
            m_callerCacheStale = false;
        }
        m_clutSnapshotStale = true;
    }
    ++m_clutSerial;
    LoadClutUnlocked(tex0, texclut, m_clut, true);
}

const std::array<uint16_t, 512> *GSCpuBackend::ClutSnapshotUnlocked(uint64_t queueIndex)
{
    if (m_clutSnapshotStale || m_clutSnapshots.empty())
    {
        if (m_clutSnapshots.empty())
        {
            m_clutSnapshots.resize(kClutSnapshots);
            m_clutSnapshotLastUse.assign(kClutSnapshots, 0u);
        }
        m_clutSnapshotSlot = (m_clutSnapshotSlot + 1u) % kClutSnapshots;
        // The slot's previous users must be drawn before it is overwritten.
        const uint64_t lastUse = m_clutSnapshotLastUse[m_clutSnapshotSlot];
        for (uint32_t i = 0; i < m_workerCount; ++i)
        {
            while (m_workerProgress[i].done.load(std::memory_order_acquire) < lastUse)
                cpuRelax();
        }
        m_clutSnapshots[m_clutSnapshotSlot] = m_clut;
        m_clutSnapshotStale = false;
    }
    m_clutSnapshotLastUse[m_clutSnapshotSlot] = queueIndex + 1u;
    return &m_clutSnapshots[m_clutSnapshotSlot];
}

void GSCpuBackend::LoadClutUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut, std::array<uint16_t, 512> &clut,
                                    bool callerCache)
{
    const bool fourBit = isFourBitIndexedPsm(tex0.psm);
    const bool sixteenBit = tex0.cpsm == GS_PSM_CT16 || tex0.cpsm == GS_PSM_CT16S;
    const bool thirtyTwoBit = tex0.cpsm == GS_PSM_CT32 || tex0.cpsm == GS_PSM_CT24;
    if (!sixteenBit && !thirtyTwoBit)
        return;

    const uint32_t entryCount = fourBit ? 16u : 256u;
    const uint32_t csaMask = sixteenBit ? 0x1Fu : 0x0Fu;
    const uint32_t destinationBase = (static_cast<uint32_t>(tex0.csa) & csaMask) << 4u;

    const bool loadCsm1Suffix = tex0.csm == 0u && thirtyTwoBit && !fourBit;
    const uint32_t firstEntry = loadCsm1Suffix ? destinationBase : 0u;

    for (uint32_t entry = firstEntry; entry < entryCount; ++entry)
    {
        uint32_t sourceX = 0u;
        uint32_t sourceY = 0u;
        uint32_t sourceWidth = 1u;

        if (tex0.csm == 0u)
        {
            const uint32_t sourceIndex = swizzleClutIndexCSM1(entry);
            sourceX = sourceIndex & 0x0Fu;
            sourceY = sourceIndex >> 4u;
        }
        else
        {
            sourceWidth = texclut.cbw != 0u ? static_cast<uint32_t>(texclut.cbw) : 1u;
            sourceX = (static_cast<uint32_t>(texclut.cou) << 4u) + entry;
            sourceY = static_cast<uint32_t>(texclut.cov);
        }

        const uint32_t raw = callerCache ? ReadTextureVramUnlocked(tex0.cpsm, tex0.cbp, sourceWidth, sourceX, sourceY)
                                         : ReadVramUnlocked(tex0.cpsm, tex0.cbp, sourceWidth, sourceX, sourceY);
        const uint32_t destination = (loadCsm1Suffix ? entry : destinationBase + entry) & (sixteenBit ? 0x1FFu : 0x0FFu);
        if (sixteenBit)
        {
            clut[destination] = static_cast<uint16_t>(raw);
        }
        else
        {
            clut[destination] = static_cast<uint16_t>(raw & 0xFFFFu);
            clut[destination + 256u] = static_cast<uint16_t>(raw >> 16u);
        }
    }
}

void GSCpuBackend::Flush()
{
    // CPU backend is immediate. GPU backends may submit command buffers here.
}

void GSCpuBackend::TextureFlush()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_texturePageCaches[0].Invalidate();
    m_decodedCaches[0].Invalidate();
    if (m_dumpRemaining != 0u)
        DumpLog("TEXFLUSH");
    if (m_workerCount == 0u || m_dumpRemaining != 0u)
        return;
    // Workers drop their cached pages when they reach this point of the queue.
    QueuedPrimitive &item = ReserveQueueSlotUnlocked();
    item.textureFlush = true;
    item.upload = false;
    item.barrier = false;
    item.writeCount = 0u;
    item.samples = false;
    PublishQueueSlotUnlocked();
}

void GSCpuBackend::Sync(GSSyncReason)
{
    // CPU backend is immediate. GPU backends may wait on fences/readbacks here.
}

uint32_t GSCpuBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    DrainUnlocked();
    return ReadVramUnlocked(psm, base, bw, x, y);
}

uint32_t GSCpuBackend::ReadVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    if (!m_vram)
        return 0u;
    return m_readVramFuncs[psm & 0x3Fu](m_vram, base, bw, x, y);
}

uint32_t GSCpuBackend::ReadTextureVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y)
{
    if (!m_vram)
        return 0u;

    return GSMem::ReadTexture(m_texturePageCaches[0], m_vram, psm, base, bw, x, y);
}

void GSCpuBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    DrainUnlocked();
    NoteDirectWrite({GSMem::PageOf(psm, base, bw, x, y) * 32u % kVramBlocks, 32u});
    WriteVramUnlocked(psm, base, bw, x, y, value);
}

void GSCpuBackend::WriteVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    if (!m_vram)
        return;
    m_writeVramFuncs[psm & 0x3Fu](m_vram, base, bw, x, y, value);
}

void GSCpuBackend::SnapshotVram(std::vector<uint8_t> &out) const
{
    // Wait for what is queued now without the lock, so the producer keeps queuing meanwhile
    // (presentation runs on its own thread); the drain under the lock then only covers what
    // was queued since.
    uint64_t queued = 0u;
    uint32_t workers = 0u;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        queued = m_queueTail.load(std::memory_order_relaxed);
        workers = m_workerCount;
    }
    for (uint32_t i = 0; i < workers; ++i)
    {
        while (m_workerProgress[i].done.load(std::memory_order_acquire) < queued)
            std::this_thread::yield();
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_dumpRemaining != 0u)
    {
        DumpLog("PRESENT (VRAM snapshot)");
        ++m_dumpPresents;
    }
    DrainUnlocked();
    if (!m_vram || m_vramSize == 0u)
    {
        out.clear();
        return;
    }
    out.resize(m_vramSize);
    std::memcpy(out.data(), m_vram, m_vramSize);
}

GSTransferSnapshot GSCpuBackend::GetTransferSnapshot() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    GSTransferSnapshot result = m_transferState;
    result.localToHostPendingBytes = m_localToHostReadPos < m_localToHostBuffer.size()
                                         ? m_localToHostBuffer.size() - m_localToHostReadPos
                                         : 0u;
    return result;
}

void GSCpuBackend::DrawPrimitive(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const auto &ctx = state.context;
    PS2_IF_AGRESSIVE_LOGS({
        const uint32_t primitiveIndex = s_debugPrimitiveCount.fetch_add(1u, std::memory_order_relaxed);
        if (primitiveIndex < 64u)
        {
            std::cout << "[gs:prim] idx=" << primitiveIndex
                      << " type=" << static_cast<uint32_t>(state.prim.type)
                      << " tme=" << static_cast<uint32_t>(state.prim.tme)
                      << " abe=" << static_cast<uint32_t>(state.prim.abe)
                      << " fst=" << static_cast<uint32_t>(state.prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(state.prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=("
                      << "tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tw=" << static_cast<uint32_t>(ctx.tex0.tw)
                      << " th=" << static_cast<uint32_t>(ctx.tex0.th)
                      << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                      << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                      << " cbp=" << ctx.tex0.cbp
                      << " cpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.cpsm) << std::dec
                      << " csm=" << static_cast<uint32_t>(ctx.tex0.csm)
                      << " csa=" << static_cast<uint32_t>(ctx.tex0.csa)
                      << ")"
                      << " texclut=("
                      << "cbw=" << static_cast<uint32_t>(state.texclut.cbw)
                      << " cou=" << static_cast<uint32_t>(state.texclut.cou)
                      << " cov=" << state.texclut.cov
                      << ")"
                      << " ofx=" << (ctx.xyoffset.ofx >> 4)
                      << " ofy=" << (ctx.xyoffset.ofy >> 4)
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test
                      << " alpha=0x" << ctx.alpha
                      << std::dec
                      << " v0=(" << batch.vertices[0].x << "," << batch.vertices[0].y << ")"
                      << " uv0=(" << (batch.vertices[0].u >> 4) << "," << (batch.vertices[0].v >> 4) << ")"
                      << " stq0=(" << batch.vertices[0].s << "," << batch.vertices[0].t << "," << batch.vertices[0].q << ")"
                      << " v1=(" << batch.vertices[1].x << "," << batch.vertices[1].y << ")"
                      << " uv1=(" << (batch.vertices[1].u >> 4) << "," << (batch.vertices[1].v >> 4) << ")"
                      << " stq1=(" << batch.vertices[1].s << "," << batch.vertices[1].t << "," << batch.vertices[1].q << ")"
                      << " v2=(" << batch.vertices[2].x << "," << batch.vertices[2].y << ")"
                      << " uv2=(" << (batch.vertices[2].u >> 4) << "," << (batch.vertices[2].v >> 4) << ")"
                      << " stq2=(" << batch.vertices[2].s << "," << batch.vertices[2].t << "," << batch.vertices[2].q << ")"
                      << " rgba0=(" << static_cast<uint32_t>(batch.vertices[0].r) << ","
                      << static_cast<uint32_t>(batch.vertices[0].g) << ","
                      << static_cast<uint32_t>(batch.vertices[0].b) << ","
                      << static_cast<uint32_t>(batch.vertices[0].a) << ")"
                      << " rgba1=(" << static_cast<uint32_t>(batch.vertices[1].r) << ","
                      << static_cast<uint32_t>(batch.vertices[1].g) << ","
                      << static_cast<uint32_t>(batch.vertices[1].b) << ","
                      << static_cast<uint32_t>(batch.vertices[1].a) << ")"
                      << " rgba2=(" << static_cast<uint32_t>(batch.vertices[2].r) << ","
                      << static_cast<uint32_t>(batch.vertices[2].g) << ","
                      << static_cast<uint32_t>(batch.vertices[2].b) << ","
                      << static_cast<uint32_t>(batch.vertices[2].a) << ")"
                      << std::endl;
        }
    });

    PS2_IF_AGRESSIVE_LOGS({
        if ((state.prim.ctxt != 0u || ctx.frame.fbp == 150u) &&
            s_debugContext1PrimitiveCount.fetch_add(1u, std::memory_order_relaxed) < 32u)
        {
            std::cout << "[gs:copy-prim]"
                      << " type=" << static_cast<uint32_t>(state.prim.type)
                      << " tme=" << static_cast<uint32_t>(state.prim.tme)
                      << " abe=" << static_cast<uint32_t>(state.prim.abe)
                      << " fst=" << static_cast<uint32_t>(state.prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(state.prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=("
                      << "tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                      << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                      << " cbp=" << ctx.tex0.cbp
                      << " cpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.cpsm) << std::dec
                      << " csm=" << static_cast<uint32_t>(ctx.tex0.csm)
                      << " csa=" << static_cast<uint32_t>(ctx.tex0.csa)
                      << ")"
                      << " texclut=("
                      << "cbw=" << static_cast<uint32_t>(state.texclut.cbw)
                      << " cou=" << static_cast<uint32_t>(state.texclut.cou)
                      << " cov=" << state.texclut.cov
                      << ")"
                      << " ofx=" << (ctx.xyoffset.ofx >> 4)
                      << " ofy=" << (ctx.xyoffset.ofy >> 4)
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test
                      << " alpha=0x" << ctx.alpha
                      << std::dec << std::endl;
        }
    });

    if (m_dumpRemaining != 0u)
        DumpPrimitiveUnlocked(batch);
    const PrimitiveBlocks blocks = primitiveBlocks(batch);
    // A primitive sampling what it draws is drawn here, in pixel order: split among workers,
    // which of its own pixels it samples would depend on their timing.
    const bool feedback = readsOwnWrites(blocks);
    if (m_workerCount == 0u || m_dumpRemaining != 0u || feedback)
    {
        if (m_workerCount != 0u)
            DrainUnlocked();
        const PixelPipeline pixel = MakePixelPipeline(state);
        TextureSampler texture = MakeTextureSampler(state);
        // Drawn in pixel order, a primitive sampling its own target must see its own writes.
        if (!feedback && state.prim.tme && primitiveArea(batch) >= kDecodedMinPixels)
        {
            texture.decoded = FindDecodedTexture(m_decodedCaches[0], texture);
            if (texture.decoded)
            {
                // Unlike the few pages of the page cache, a whole decoded texture would outlive
                // writes made without a TEXFLUSH.
                const auto &tex0 = state.context.tex0;
                const auto pages = rowBlocks(tex0.tbp0, tex0.tbw, tex0.psm, 0, std::max(texture.texH, 1) - 1);
                const uint32_t firstPage = pages.first / 32u;
                const uint32_t count = std::min<uint32_t>((pages.first % 32u + pages.second + 31u) / 32u, kVramBlocks / 32u);
                bool stale = false;
                for (uint32_t i = 0; i < count && !stale; ++i)
                    stale = m_pageWriteClock[(firstPage + i) % (kVramBlocks / 32u)] > texture.decoded->writeClock;
                if (stale)
                    std::fill(texture.decoded->tileReady.begin(), texture.decoded->tileReady.end(), 0u);
                texture.decoded->writeClock = m_writeClock;
            }
        }
        DrawWith(batch, RasterContext{&pixel, &texture, &m_texturePageCaches[0], 0u, 1u});
        NoteDirectWrite(blocks.frame);
        if (blocks.writesZ)
            NoteDirectWrite(blocks.z);
        if (m_dumpRemaining != 0u && --m_dumpRemaining == 0u)
            DumpFinishUnlocked();
        return;
    }

    // Rather than stopping the producer, the workers wait for each other at this primitive when
    // it samples what queued primitives draw (render to texture), draws over what they still
    // sample, or mixes pixel and depth writes on the same pages (their owners differ). Depth
    // words follow the owner of their color page, so a Z buffer reused under another color-page
    // layout (double buffering) also gets a barrier.
    bool barrier = m_barrierRequested ||
                   (blocks.samples && (overlapsAny(blocks.texture, m_pixelWritesSinceBarrier) ||
                                       overlapsAny(blocks.texture, m_zWritesSinceBarrier))) ||
                   overlapsAny(blocks.frame, m_readsSinceBarrier) || overlapsAny(blocks.frame, m_zWritesSinceBarrier) ||
                   (blocks.writesZ && (overlapsAny(blocks.z, m_readsSinceBarrier) ||
                                       overlapsAny(blocks.z, m_pixelWritesSinceBarrier)));
    uint64_t zOwnerKey = 0;
    const bool writesZ = blocks.writesZ;
    if (writesZ)
    {
        const auto &frame = state.context.frame;
        uint32_t pageWidth = 0;
        uint32_t pageHeight = 0;
        pageGeometry(frame.psm, pageWidth, pageHeight);
        const uint32_t basePage = GSInternal::framePageBaseToBlock(frame.fbp) / 32u;
        const uint32_t pagesPerRow = std::max(1u, (std::max<uint32_t>(frame.fbw, 1u) * 64u + pageWidth - 1u) / pageWidth);
        zOwnerKey = (static_cast<uint64_t>(state.context.zbuf.zbp) << 40u) | (static_cast<uint64_t>(basePage % m_workerCount) << 32u) |
                    (static_cast<uint64_t>(pagesPerRow) << 16u) | (pageWidth << 8u) | pageHeight;
        for (const uint64_t key : m_zOwnerKeys)
            barrier = barrier || ((key >> 40u) == state.context.zbuf.zbp && key != zOwnerKey);
    }
    if (barrier)
    {
        m_pixelWritesSinceBarrier.clear();
        m_zWritesSinceBarrier.clear();
        m_readsSinceBarrier.clear();
        m_zOwnerKeys.clear();
        m_barrierRequested = false;
    }
    if (writesZ && std::find(m_zOwnerKeys.begin(), m_zOwnerKeys.end(), zOwnerKey) == m_zOwnerKeys.end())
        m_zOwnerKeys.push_back(zOwnerKey);

    QueuedPrimitive &item = ReserveQueueSlotUnlocked();
    item.batch = batch;
    item.pixel = MakePixelPipeline(state);
    item.texture = MakeTextureSampler(state);
    if (state.prim.tme && (isFourBitIndexedPsm(state.context.tex0.psm) || isEightBitIndexedPsm(state.context.tex0.psm)))
        item.texture.clut = ClutSnapshotUnlocked(m_queueTail.load(std::memory_order_relaxed));
    item.textureFlush = false;
    item.upload = false;
    item.barrier = barrier;
    item.area = primitiveArea(batch);
    item.writes[0] = blocks.frame;
    item.writes[1] = blocks.z;
    item.writeCount = blocks.writesZ ? 2u : 1u;
    // Workers decode textures by 32x32 tiles, so they may read rows of it the primitive never samples.
    if (blocks.samples)
        item.reads = rowBlocks(ctx.tex0.tbp0, ctx.tex0.tbw, ctx.tex0.psm, 0,
                               std::min(2047, (1 << std::min<uint32_t>(ctx.tex0.th, 10u)) - 1));
    item.samples = blocks.samples;
    NotePendingRange(m_pendingWrites, blocks.frame);
    NoteBarrierRange(m_pixelWritesSinceBarrier, blocks.frame);
    if (blocks.writesZ)
    {
        NotePendingRange(m_pendingWrites, blocks.z);
        NoteBarrierRange(m_zWritesSinceBarrier, blocks.z);
    }
    if (blocks.samples)
        NoteBarrierRange(m_readsSinceBarrier, blocks.texture);
    PublishQueueSlotUnlocked();
}

bool GSCpuBackend::OverlapsInFlightUnlocked(std::pair<uint32_t, uint32_t> range, bool includeReads) const
{
    const uint64_t tail = m_queueTail.load(std::memory_order_relaxed);
    uint64_t first = tail;
    for (uint32_t i = 0; i < m_workerCount; ++i)
        first = std::min(first, m_workerProgress[i].done.load(std::memory_order_acquire));
    for (uint64_t index = first; index < tail; ++index)
    {
        const QueuedPrimitive &item = m_queue[index & (kQueueSize - 1u)];
        for (uint8_t w = 0; w < item.writeCount; ++w)
            if (blocksOverlap(range, item.writes[w]))
                return true;
        if (includeReads && item.samples && blocksOverlap(range, item.reads))
            return true;
    }
    return false;
}

GSCpuBackend::QueuedPrimitive &GSCpuBackend::ReserveQueueSlotUnlocked()
{
    const uint64_t tail = m_queueTail.load(std::memory_order_relaxed);
    for (uint32_t i = 0; i < m_workerCount; ++i)
    {
        // Ring full: wait for the slowest worker to free the slot.
        while (tail - m_workerProgress[i].done.load(std::memory_order_acquire) >= kQueueSize)
            cpuRelax();
    }
    return m_queue[tail & (kQueueSize - 1u)];
}

void GSCpuBackend::PublishQueueSlotUnlocked()
{
    m_callerCacheStale = true;
    m_queueTail.store(m_queueTail.load(std::memory_order_relaxed) + 1u, std::memory_order_seq_cst);
    if (m_sleepingWorkers.load(std::memory_order_seq_cst) != 0u)
    {
        std::lock_guard<std::mutex> lock(m_workMutex);
        m_workCv.notify_all();
    }
}

void GSCpuBackend::DrawQueued(const QueuedPrimitive &item, uint32_t index)
{
    if (item.textureFlush)
    {
        m_texturePageCaches[index + 1u].Invalidate();
        m_decodedCaches[index + 1u].Invalidate();
        return;
    }
    if (item.upload)
    {
        const RasterContext rows{nullptr, nullptr, nullptr, index, m_workerCount};
        WriteUploadPixels(item.transfer, item.uploadFirstPixel, item.uploadTotalPixels, item.uploadData.data(),
                          static_cast<uint32_t>(item.uploadData.size()), &rows, true);
        return;
    }
    const GSPrimitiveBatch &batch = item.batch;
    const GSDrawState &state = batch.state;
    TextureSampler texture = item.texture;
    texture.state = &state;
    if (state.prim.tme && item.area >= kDecodedMinPixels)
        texture.decoded = FindDecodedTexture(m_decodedCaches[index + 1u], texture);
    RasterContext raster{&item.pixel, &texture, &m_texturePageCaches[index + 1u], index, m_workerCount};
    const auto &frame = state.context.frame;
    pageGeometry(frame.psm, raster.pageWidth, raster.pageHeight);
    raster.basePage = GSInternal::framePageBaseToBlock(frame.fbp) / 32u;
    raster.pagesPerRow = std::max(1u, (std::max<uint32_t>(frame.fbw, 1u) * 64u + raster.pageWidth - 1u) / raster.pageWidth);
    DrawWith(batch, raster);
}

void GSCpuBackend::DrawWith(const GSPrimitiveBatch &batch, const RasterContext &raster)
{
    const GSDrawState &state = batch.state;
    switch (state.prim.type)
    {
    case GS_PRIM_SPRITE:
        DrawSprite(batch, raster);
        break;
    case GS_PRIM_TRIANGLE:
    case GS_PRIM_TRISTRIP:
    case GS_PRIM_TRIFAN:
        DrawTriangle(batch, raster);
        break;
    case GS_PRIM_LINE:
    case GS_PRIM_LINESTRIP:
        DrawLine(batch, raster);
        break;
    case GS_PRIM_POINT:
    {
        const GSVertex &v = batch.vertices[0];
        const auto &ctx = state.context;
        int px = static_cast<int>(v.x) - (ctx.xyoffset.ofx >> 4);
        int py = static_cast<int>(v.y) - (ctx.xyoffset.ofy >> 4);
        if (raster.owns(px, py))
            WritePixel(*raster.pixel, px, py, static_cast<u32>(v.z), v.r, v.g, v.b, v.a, v.fog);
        break;
    }
    default:
        break;
    }
}

std::atomic<bool> GSCpuBackend::s_asyncRasterization{false};

void GSCpuBackend::SetAsyncRasterization(bool enabled) noexcept
{
    s_asyncRasterization.store(enabled, std::memory_order_relaxed);
}

void GSCpuBackend::StartWorkersUnlocked()
{
    if (m_workerCount != 0u)
        return;
    // Leave cores for the EE, the render thread and the IOP; at PS2 resolutions more than
    // 8 stripes buys little.
    const uint32_t hardware = std::max(1u, std::thread::hardware_concurrency());
    m_workerCount = std::clamp(hardware > 4u ? hardware - 4u : 1u, 1u, 8u);
    m_texturePageCaches.resize(m_workerCount + 1u);
    m_decodedCaches.resize(m_workerCount + 1u);
    m_queue.resize(kQueueSize);
    m_workerProgress = std::make_unique<WorkerProgress[]>(m_workerCount);
    for (uint32_t i = 0; i < m_workerCount; ++i)
        m_workers.emplace_back([this, i]() { WorkerMain(i); });
    std::fprintf(stderr, "[gs] asynchronous rasterization on %u worker threads\n", m_workerCount);
}

namespace
{
    std::atomic<uint64_t> g_drainNanoseconds{0};
    std::atomic<uint64_t> g_drainCount{0};
}

std::pair<uint64_t, uint64_t> GSCpuBackend::TakeDrainStats() noexcept
{
    return {g_drainNanoseconds.exchange(0u), g_drainCount.exchange(0u)};
}

void GSCpuBackend::DrainUnlocked() const
{
    const auto start = std::chrono::steady_clock::now();
    const uint64_t tail = m_queueTail.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < m_workerCount; ++i)
    {
        while (m_workerProgress[i].done.load(std::memory_order_acquire) < tail)
            cpuRelax();
    }
    g_drainNanoseconds.fetch_add(static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()));
    g_drainCount.fetch_add(1u);
    // The write lists stay: workers' decoded textures may predate those writes, and the next
    // primitive sampling them needs the barrier that refreshes them.
    m_pendingWrites.clear();
    m_readsSinceBarrier.clear();
    m_zOwnerKeys.clear();
}

void GSCpuBackend::NoteDirectWrite(std::pair<uint32_t, uint32_t> blocks)
{
    ++m_writeClock;
    const uint32_t firstPage = blocks.first / 32u;
    const uint32_t pages = std::min<uint32_t>((blocks.first % 32u + blocks.second + 31u) / 32u, kVramBlocks / 32u);
    for (uint32_t i = 0; i < pages; ++i)
        m_pageWriteClock[(firstPage + i) % (kVramBlocks / 32u)] = m_writeClock;
    if (m_workerCount != 0u)
        NoteBarrierRange(m_pixelWritesSinceBarrier, blocks);
}

void GSCpuBackend::NotePendingRange(std::vector<std::pair<uint32_t, uint32_t>> &ranges, std::pair<uint32_t, uint32_t> range)
{
    // Merge with an overlapping or adjacent range (non-wrapping ones) to keep the list short.
    for (auto &pending : ranges)
    {
        const uint64_t end = static_cast<uint64_t>(pending.first) + pending.second;
        const uint64_t rangeEnd = static_cast<uint64_t>(range.first) + range.second;
        if (end <= kVramBlocks && rangeEnd <= kVramBlocks && range.first <= end && pending.first <= rangeEnd)
        {
            const uint32_t start = std::min(pending.first, range.first);
            pending = {start, static_cast<uint32_t>(std::max(end, rangeEnd) - start)};
            return;
        }
    }
    if (ranges.size() >= kMaxPendingWrites)
        DrainUnlocked(); // clears both lists
    ranges.push_back(range);
}

void GSCpuBackend::NoteBarrierRange(std::vector<std::pair<uint32_t, uint32_t>> &ranges, std::pair<uint32_t, uint32_t> range)
{
    for (auto &pending : ranges)
    {
        const uint64_t end = static_cast<uint64_t>(pending.first) + pending.second;
        const uint64_t rangeEnd = static_cast<uint64_t>(range.first) + range.second;
        if (end <= kVramBlocks && rangeEnd <= kVramBlocks && range.first <= end && pending.first <= rangeEnd)
        {
            const uint32_t start = std::min(pending.first, range.first);
            pending = {start, static_cast<uint32_t>(std::max(end, rangeEnd) - start)};
            return;
        }
    }
    if (ranges.size() >= kMaxPendingWrites)
    {
        // Too many ranges to track: the next item waits for everything before it.
        m_pixelWritesSinceBarrier.clear();
        m_zWritesSinceBarrier.clear();
        m_readsSinceBarrier.clear();
        m_barrierRequested = true;
        return;
    }
    ranges.push_back(range);
}

void GSCpuBackend::WorkerMain(uint32_t index)
{
    ThreadNaming::SetCurrentThreadName("GSRaster");
    uint64_t next = 0u;
    for (;;)
    {
        // Spin briefly (primitives of a frame arrive back to back), then sleep.
        uint64_t tail = m_queueTail.load(std::memory_order_acquire);
        for (uint32_t spin = 0; tail == next && spin < kWorkerSpinIterations; ++spin)
        {
            cpuRelax();
            tail = m_queueTail.load(std::memory_order_acquire);
        }
        if (tail == next)
        {
            std::unique_lock<std::mutex> lock(m_workMutex);
            m_sleepingWorkers.fetch_add(1u, std::memory_order_seq_cst);
            m_workCv.wait(lock, [&]()
                          { return m_stopWorkers.load(std::memory_order_acquire) ||
                                   m_queueTail.load(std::memory_order_seq_cst) != next; });
            m_sleepingWorkers.fetch_sub(1u, std::memory_order_seq_cst);
            tail = m_queueTail.load(std::memory_order_acquire);
        }
        if (tail == next && m_stopWorkers.load(std::memory_order_acquire))
            return;
        for (; next < tail; ++next)
        {
            const QueuedPrimitive &item = m_queue[next & (kQueueSize - 1u)];
            if (item.barrier)
            {
                for (uint32_t other = 0; other < m_workerCount; ++other)
                {
                    while (m_workerProgress[other].done.load(std::memory_order_acquire) < next)
                        cpuRelax();
                }
                // It samples what earlier primitives drew: cached pages and decoded copies may be
                // out of date. (The immediate rasterizer's single page cache keeps written pages
                // hidden until TEXFLUSH, like the GS texture buffer; a worker reads few enough
                // pages that its cache would keep them far longer, so workers refresh here.)
                m_texturePageCaches[index + 1u].Invalidate();
                m_decodedCaches[index + 1u].Invalidate();
            }
            DrawQueued(item, index);
            m_workerProgress[index].done.store(next + 1u, std::memory_order_release);
        }
    }
}

void GSCpuBackend::WritePixel(const PixelPipeline &p, int x, int y, uint32_t z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog)
{
    if (x < p.scissorX0 || x > p.scissorX1 || y < p.scissorY0 || y > p.scissorY1)
        return;

    if (p.fge)
    {
        const uint32_t inverseFog = 255u - fog;
        auto applyFog = [&](uint8_t input, uint8_t fogColor) -> uint8_t
        {
            return static_cast<uint8_t>(((static_cast<uint32_t>(fog) * input) >> 8) + ((inverseFog * fogColor) >> 8));
        };

        r = applyFog(r, p.fogR);
        g = applyFog(g, p.fogG);
        b = applyFog(b, p.fogB);
    }

    PixelWriteMask writeMask{};
    bool preserveDestinationAlpha = false;
    // small optimization, avoid reading the framebuffer for simple draws
    bool frmw = p.passReadsFramebuffer;
    if (p.alphaTest)
    {
        writeMask = classifyAlphaTest(p.test, a, p.fpsm);
        if (!writeMask.writesAnything())
        {
            return;
        }
        preserveDestinationAlpha = writeMask.writeRgb && !writeMask.writeAlpha && p.fpsm == GS_PSM_CT32;
        frmw = p.dateNeedsRead || (writeMask.writesFramebuffer() && (p.fbmsk != 0u || p.abe || preserveDestinationAlpha));
    }

    u32 rawFramebufferPixel = 0;
    u32 fbrgba = 0;
    if (frmw)
    {
        rawFramebufferPixel = p.readFrame(m_vram, p.fbp, p.fbw, x, y);
        fbrgba = rawFramebufferPixel;

        if (p.frame16)
        {
            fbrgba = Rgba5551ToRgba8888(fbrgba);
        }
        else if (p.frame24)
        {
            // The GS supplies 0x80 as destination alpha for RGB24 blending.
            fbrgba |= 0x80000000u;
        }
    }

    if (!passesDestinationAlphaTest(p.test, p.fpsm, rawFramebufferPixel))
    {
        return;
    }

    switch (p.ztst)
    {
    case 0:
        return;
    case 1:
        break;
    case 2:
        if (z < p.readZ(m_vram, p.zbp, p.fbw, x, y))
            return;
        break;
    case 3:
        if (z <= p.readZ(m_vram, p.zbp, p.fbw, x, y))
            return;
        break;
    }

    if (writeMask.writesFramebuffer())
    {
        // PABE disables alpha blending when the source alpha MSB is clear.
        if (p.abe && !(p.pabe && (a & 0x80u) == 0u))
        {
            const int dr = static_cast<int>(fbrgba & 0xFF);
            const int dg = static_cast<int>((fbrgba >> 8) & 0xFF);
            const int db = static_cast<int>((fbrgba >> 16) & 0xFF);
            const int da = static_cast<int>((fbrgba >> 24) & 0xFF);

            auto pickRGB = [](uint8_t sel, int cs, int cd) -> int
            {
                if (sel == 0)
                    return cs;
                if (sel == 1)
                    return cd;
                return 0;
            };
            const int cAlpha = (p.csel == 0) ? a : (p.csel == 1) ? da
                                                                 : p.fix;

            r = clampU8(((pickRGB(p.asel, r, dr) - pickRGB(p.bsel, r, dr)) * cAlpha >> 7) + pickRGB(p.dsel, r, dr));
            g = clampU8(((pickRGB(p.asel, g, dg) - pickRGB(p.bsel, g, dg)) * cAlpha >> 7) + pickRGB(p.dsel, g, dg));
            b = clampU8(((pickRGB(p.asel, b, db) - pickRGB(p.bsel, b, db)) * cAlpha >> 7) + pickRGB(p.dsel, b, db));
        }

        if (writeMask.writeAlpha && p.fba)
        {
            a = static_cast<uint8_t>(a | 0x80u);
        }

        u32 pixel = pack32(r, g, b, a);

        if (p.fbmsk != 0)
        {
            pixel = (pixel & ~p.fbmsk) | (fbrgba & p.fbmsk);
        }

        if (preserveDestinationAlpha)
        {
            pixel = (pixel & 0x00FFFFFFu) | (fbrgba & 0xFF000000u);
        }

        // format conversion
        if (p.frame16)
        {
            pixel = Rgba8888ToRgba5551(pixel);
        }

        p.writeFrame(m_vram, p.fbp, p.fbw, x, y, pixel);
    }

    if (writeMask.writeDepth && !p.zmask)
    {
        p.writeZ(m_vram, p.zbp, p.fbw, x, y, z);
    }
}

uint32_t GSCpuBackend::LookupCLUT(const GSDrawState &state,
                                  const std::array<uint16_t, 512> &clut,
                                  uint8_t index,
                                  uint8_t cpsm,
                                  uint8_t csm,
                                  uint8_t csa,
                                  uint8_t sourcePsm)
{
    const bool sixteenBit = cpsm == GS_PSM_CT16 || cpsm == GS_PSM_CT16S;
    const uint32_t csaMask = sixteenBit ? 0x1Fu : 0x0Fu;
    const uint32_t clutBase = (static_cast<uint32_t>(csa) & csaMask) << 4u;
    const uint32_t sourceIndex = isFourBitIndexedPsm(sourcePsm)
                                     ? (static_cast<uint32_t>(index) & 0x0Fu)
                                     : static_cast<uint32_t>(index);

    uint32_t clutIndex = (clutBase + sourceIndex) & (sixteenBit ? 0x1FFu : 0x0FFu);
    if (!sixteenBit && csm == 0u && isEightBitIndexedPsm(sourcePsm))
    {
        const uint32_t block = std::min((sourceIndex & 0xF0u) + clutBase, 240u);
        clutIndex = block + (sourceIndex & 0x0Fu);
    }

    switch (cpsm)
    {
    case GS_PSM_CT32:
    {
        const uint32_t raw = static_cast<uint32_t>(clut[clutIndex]) | (static_cast<uint32_t>(clut[clutIndex + 256u]) << 16u);
        return applyTexa(state.texa, cpsm, raw);
    }
    case GS_PSM_CT24:
    {
        const uint32_t raw = static_cast<uint32_t>(clut[clutIndex]) | (static_cast<uint32_t>(clut[clutIndex + 256u]) << 16u);
        return applyTexa(state.texa, cpsm, raw & 0x00FFFFFFu);
    }
    case GS_PSM_CT16:
    case GS_PSM_CT16S:
        return applyTexa(state.texa, cpsm, Rgba5551ToRgba8888(clut[clutIndex]));
    default:
        break;
    }

    return 0xFFFF00FFu;
}

uint32_t GSCpuBackend::FetchTexel(const TextureSampler &sampler, GSMem::TexturePageCache &cache, int u, int v)
{
    const GSDrawState &state = *sampler.state;
    const auto &tex = state.context.tex0;
    u32 out = GSMem::ReadTexture(cache, m_vram, tex.psm, tex.tbp0, tex.tbw, u, v);

    switch (tex.psm)
    {
    case GS_PSM_CT32:
    case GS_PSM_Z32:
    case GS_PSM_CT24:
    case GS_PSM_Z24:
        return applyTexa(state.texa, tex.psm, out);
    case GS_PSM_CT16:
    case GS_PSM_CT16S:
    case GS_PSM_Z16:
    case GS_PSM_Z16S:
        return applyTexa(state.texa, tex.psm, Rgba5551ToRgba8888(out));
    case GS_PSM_T8:
    case GS_PSM_T8H:
    case GS_PSM_T4:
    case GS_PSM_T4HL:
    case GS_PSM_T4HH:
        return LookupCLUT(state, *sampler.clut, static_cast<u8>(out), tex.cpsm, tex.csm, tex.csa, tex.psm);
    }

    return 0xFFFF00FFu;
}

void GSCpuBackend::DecodeTile(DecodedTexture &texture, const TextureSampler &sampler, GSMem::TexturePageCache &cache, uint32_t tile)
{
    const int u0 = static_cast<int>(tile % texture.tilesPerRow) << kDecodedTileShift;
    const int v0 = static_cast<int>(tile / texture.tilesPerRow) << kDecodedTileShift;
    const int u1 = std::min(u0 + (1 << kDecodedTileShift), texture.width);
    const int v1 = std::min(v0 + (1 << kDecodedTileShift), texture.height);
    for (int v = v0; v < v1; ++v)
    {
        uint32_t *row = texture.texels.get() + static_cast<size_t>(v) * static_cast<size_t>(texture.width);
        for (int u = u0; u < u1; ++u)
            row[u] = FetchTexel(sampler, cache, u, v);
    }
    texture.tileReady[tile] = 1u;
}

GSCpuBackend::DecodedTexture *GSCpuBackend::FindDecodedTexture(DecodedTextureCache &cache, const TextureSampler &sampler)
{
    const GSDrawState &state = *sampler.state;
    const auto &tex = state.context.tex0;
    if (sampler.texW * sampler.texH > kDecodedMaxTexels)
        return nullptr;
    const bool indexed = isFourBitIndexedPsm(tex.psm) || isEightBitIndexedPsm(tex.psm);
    auto matches = [&](const DecodedTexture &t)
    {
        return t.tbp0 == tex.tbp0 && t.tbw == tex.tbw && t.psm == tex.psm && t.width == sampler.texW &&
               t.height == sampler.texH && t.texa.ta0 == state.texa.ta0 && t.texa.ta1 == state.texa.ta1 &&
               t.texa.aem == state.texa.aem &&
               (!indexed || (t.cpsm == tex.cpsm && t.csm == tex.csm && t.csa == tex.csa && t.clutSerial == sampler.clutSerial));
    };
    ++cache.clock;
    for (DecodedTexture &t : cache.textures)
    {
        if (t.valid && matches(t))
        {
            t.lastUse = cache.clock;
            return &t;
        }
    }
    // Reuse an invalidated entry, else the least recently used one.
    DecodedTexture *slot = nullptr;
    if (cache.textures.size() < kDecodedTexturesPerWorker)
    {
        auto free = std::find_if(cache.textures.begin(), cache.textures.end(), [](const DecodedTexture &t) { return !t.valid; });
        if (free != cache.textures.end())
        {
            slot = &*free;
        }
        else
        {
            cache.textures.reserve(kDecodedTexturesPerWorker);
            slot = &cache.textures.emplace_back();
        }
    }
    else
    {
        slot = &*std::min_element(cache.textures.begin(), cache.textures.end(), [](const DecodedTexture &a, const DecodedTexture &b)
                                  { return (a.valid ? a.lastUse : 0u) < (b.valid ? b.lastUse : 0u); });
    }
    DecodedTexture &t = *slot;
    t.valid = true;
    t.tbp0 = tex.tbp0;
    t.tbw = tex.tbw;
    t.psm = tex.psm;
    t.cpsm = tex.cpsm;
    t.csm = tex.csm;
    t.csa = tex.csa;
    t.width = sampler.texW;
    t.height = sampler.texH;
    t.clutSerial = sampler.clutSerial;
    t.texa = state.texa;
    t.lastUse = cache.clock;
    t.tilesPerRow = static_cast<uint32_t>((sampler.texW + (1 << kDecodedTileShift) - 1) >> kDecodedTileShift);
    const uint32_t tileRows = static_cast<uint32_t>((sampler.texH + (1 << kDecodedTileShift) - 1) >> kDecodedTileShift);
    const size_t texelCount = static_cast<size_t>(sampler.texW) * static_cast<size_t>(sampler.texH);
    if (t.texelCapacity < texelCount)
    {
        t.texels.reset(new uint32_t[texelCount]);
        t.texelCapacity = texelCount;
    }
    t.tileReady.assign(static_cast<size_t>(t.tilesPerRow) * tileRows, 0u);
    return &t;
}

uint32_t GSCpuBackend::SampleTexture(const TextureSampler &sampler, GSMem::TexturePageCache &cache, float s, float t, float q, uint16_t u, uint16_t v)
{
    const GSDrawState &state = *sampler.state;
    const auto &ctx = state.context;
    const auto &tex = ctx.tex0;

    const int texW = sampler.texW;
    const int texH = sampler.texH;

    float texUf, texVf;
    if (sampler.fst)
    {
        texUf = static_cast<float>(u) / 16.0f;
        texVf = static_cast<float>(v) / 16.0f;
    }
    else
    {
        const float invQ = 1.0f / fabsQ(q);
        texUf = s * invQ * static_cast<float>(texW);
        texVf = t * invQ * static_cast<float>(texH);
    }

    auto samplePoint = [&](int sampleU, int sampleV) -> uint32_t
    {
        sampleU = wrapTextureCoordinate(sampleU, texW, sampler.wrapU, sampler.minU, sampler.maxU);
        sampleV = wrapTextureCoordinate(sampleV, texH, sampler.wrapV, sampler.minV, sampler.maxV);

        if (DecodedTexture *decoded = sampler.decoded)
        {
            // Region modes can address texels outside the texture rectangle.
            if (static_cast<unsigned>(sampleU) < static_cast<unsigned>(decoded->width) &&
                static_cast<unsigned>(sampleV) < static_cast<unsigned>(decoded->height))
            {
                const uint32_t tile = static_cast<uint32_t>(sampleV >> kDecodedTileShift) * decoded->tilesPerRow +
                                      static_cast<uint32_t>(sampleU >> kDecodedTileShift);
                if (!decoded->tileReady[tile])
                    DecodeTile(*decoded, sampler, cache, tile);
                return decoded->texels[static_cast<size_t>(sampleV) * static_cast<size_t>(decoded->width) + static_cast<size_t>(sampleU)];
            }
        }
        return FetchTexel(sampler, cache, sampleU, sampleV);
    };

    if (!sampler.linear)
    {
        return samplePoint(static_cast<int>(texUf), static_cast<int>(texVf));
    }

    const float sampleU = texUf - 0.5f;
    const float sampleV = texVf - 0.5f;
    const int u0 = floorToInt(sampleU);
    const int v0 = floorToInt(sampleV);
    const int u1 = u0 + 1;
    const int v1 = v0 + 1;
    const int fx = static_cast<int>((sampleU - static_cast<float>(u0)) * 256.0f + 0.5f);
    const int fy = static_cast<int>((sampleV - static_cast<float>(v0)) * 256.0f + 0.5f);
    if ((fx | fy) == 0)
    {
        return samplePoint(u0, v0); // texel-aligned (e.g. 1:1 movie frames): the blend is c00
    }

    const uint32_t c00 = samplePoint(u0, v0);
    const uint32_t c10 = samplePoint(u1, v0);
    const uint32_t c01 = samplePoint(u0, v1);
    const uint32_t c11 = samplePoint(u1, v1);

    const uint8_t r = lerpChannel(static_cast<uint8_t>(c00 & 0xFFu),
                                  static_cast<uint8_t>(c10 & 0xFFu),
                                  static_cast<uint8_t>(c01 & 0xFFu),
                                  static_cast<uint8_t>(c11 & 0xFFu),
                                  fx, fy);
    const uint8_t g = lerpChannel(static_cast<uint8_t>((c00 >> 8) & 0xFFu),
                                  static_cast<uint8_t>((c10 >> 8) & 0xFFu),
                                  static_cast<uint8_t>((c01 >> 8) & 0xFFu),
                                  static_cast<uint8_t>((c11 >> 8) & 0xFFu),
                                  fx, fy);
    const uint8_t b = lerpChannel(static_cast<uint8_t>((c00 >> 16) & 0xFFu),
                                  static_cast<uint8_t>((c10 >> 16) & 0xFFu),
                                  static_cast<uint8_t>((c01 >> 16) & 0xFFu),
                                  static_cast<uint8_t>((c11 >> 16) & 0xFFu),
                                  fx, fy);
    const uint8_t a = lerpChannel(static_cast<uint8_t>((c00 >> 24) & 0xFFu),
                                  static_cast<uint8_t>((c10 >> 24) & 0xFFu),
                                  static_cast<uint8_t>((c01 >> 24) & 0xFFu),
                                  static_cast<uint8_t>((c11 >> 24) & 0xFFu),
                                  fx, fy);

    return static_cast<uint32_t>(r) |
           (static_cast<uint32_t>(g) << 8) |
           (static_cast<uint32_t>(b) << 16) |
           (static_cast<uint32_t>(a) << 24);
}

void GSCpuBackend::DrawSprite(const GSPrimitiveBatch &batch, const RasterContext &raster)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const auto &ctx = state.context;

    float fx0 = v0.x - static_cast<float>(ctx.xyoffset.ofx) / 16.0f;
    float fy0 = v0.y - static_cast<float>(ctx.xyoffset.ofy) / 16.0f;
    float fx1 = v1.x - static_cast<float>(ctx.xyoffset.ofx) / 16.0f;
    float fy1 = v1.y - static_cast<float>(ctx.xyoffset.ofy) / 16.0f;
    u32 z1 = static_cast<u32>(v1.z);

    if (fx0 > fx1)
        std::swap(fx0, fx1);
    if (fy0 > fy1)
        std::swap(fy0, fy1);

    // The GS covers the pixels whose integer coordinates satisfy x0 <= x < x1 (and the same for
    // y), so a sprite ending at 63.5 includes pixel 63.
    const int unclippedX0 = static_cast<int>(std::ceil(fx0));
    const int unclippedY0 = static_cast<int>(std::ceil(fy0));
    const int unclippedX1 = std::max(unclippedX0, static_cast<int>(std::ceil(fx1)) - 1);
    const int unclippedY1 = std::max(unclippedY0, static_cast<int>(std::ceil(fy1)) - 1);
    const int spanX = unclippedX1 - unclippedX0 + 1;
    const int spanY = unclippedY1 - unclippedY0 + 1;

    // If the sprite rectangle is fully outside scissor, nothing should render.
    if (unclippedX1 < ctx.scissor.x0 || unclippedX0 > ctx.scissor.x1 ||
        unclippedY1 < ctx.scissor.y0 || unclippedY0 > ctx.scissor.y1)
        return;

    const int drawX0 = clampInt(unclippedX0, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY0 = clampInt(unclippedY0, ctx.scissor.y0, ctx.scissor.y1);
    const int drawX1 = clampInt(unclippedX1, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY1 = clampInt(unclippedY1, ctx.scissor.y0, ctx.scissor.y1);

    const uint64_t alphaReg = ctx.alpha;
    const uint8_t alphaMode = static_cast<uint8_t>(alphaReg & 0xFFu);
    const uint8_t alphaFix = static_cast<uint8_t>((alphaReg >> 32) & 0xFFu);

    uint8_t r = v1.r, g = v1.g, b = v1.b, a = v1.a;

    if (state.prim.tme)
    {
        const auto &tex = ctx.tex0;
        const int texW = state.textureWidth;
        const int texH = state.textureHeight;

        float u0f, v0f, u1f, v1f;
        if (state.prim.fst)
        {
            // UV registers are 12.4 fixed point; keep the fraction (games use sub-texel offsets,
            // e.g. per-field half-texel shifts when copying an interlaced frame with bilinear filtering).
            u0f = static_cast<float>(v0.u) / 16.0f;
            v0f = static_cast<float>(v0.v) / 16.0f;
            u1f = static_cast<float>(v1.u) / 16.0f;
            v1f = static_cast<float>(v1.v) / 16.0f;
        }
        else
        {
            const float q0 = fabsQ(v0.q);
            const float q1 = fabsQ(v1.q);
            u0f = (v0.s / q0) * static_cast<float>(texW);
            v0f = (v0.t / q0) * static_cast<float>(texH);
            u1f = (v1.s / q1) * static_cast<float>(texW);
            v1f = (v1.t / q1) * static_cast<float>(texH);
        }

        float spriteW = static_cast<float>(spanX);
        float spriteH = static_cast<float>(spanY);
        if (spriteW < 1.0f)
            spriteW = 1.0f;
        if (spriteH < 1.0f)
            spriteH = 1.0f;

        for (int y = drawY0; y <= drawY1; ++y)
        {
            float ty = (static_cast<float>(y - unclippedY0) + 0.5f) / spriteH;
            float texVf = v0f + (v1f - v0f) * ty;

            for (int runStart = drawX0, runEnd = 0; raster.ownedRun(y, runStart, runEnd, drawX1); runStart = runEnd + 1)
            for (int x = runStart; x <= runEnd; ++x)
            {
                float tx = (static_cast<float>(x - unclippedX0) + 0.5f) / spriteW;
                float texUf = u0f + (u1f - u0f) * tx;
                uint32_t texel = 0xFFFF00FFu;
                if (state.prim.fst)
                {
                    const int fixedU = static_cast<int>((texUf * 16.0f) + 0.5f);
                    const int fixedV = static_cast<int>((texVf * 16.0f) + 0.5f);
                    const uint16_t sampleU = static_cast<uint16_t>(clampInt(fixedU, 0, 0xFFFF));
                    const uint16_t sampleV = static_cast<uint16_t>(clampInt(fixedV, 0, 0xFFFF));
                    texel = SampleTexture(*raster.texture, *raster.cache, 0.0f, 0.0f, 1.0f, sampleU, sampleV);
                }
                else
                {
                    texel = SampleTexture(*raster.texture, *raster.cache, texUf / static_cast<float>(texW), texVf / static_cast<float>(texH), 1.0f, 0u, 0u);
                }

                uint8_t tr = static_cast<uint8_t>(texel & 0xFF);
                uint8_t tg = static_cast<uint8_t>((texel >> 8) & 0xFF);
                uint8_t tb = static_cast<uint8_t>((texel >> 16) & 0xFF);
                uint8_t ta = static_cast<uint8_t>((texel >> 24) & 0xFF);

                const TextureCombineResult color = combineTexture(tex, r, g, b, a, tr, tg, tb, ta);
                WritePixel(*raster.pixel, x, y, z1, color.r, color.g, color.b, color.a, v1.fog);
            }
        }
    }
    else
    {
        for (int y = drawY0; y <= drawY1; ++y)
        {
            for (int runStart = drawX0, runEnd = 0; raster.ownedRun(y, runStart, runEnd, drawX1); runStart = runEnd + 1)
                for (int x = runStart; x <= runEnd; ++x)
                    WritePixel(*raster.pixel, x, y, z1, r, g, b, a, v1.fog);
        }
    }
}

void GSCpuBackend::DrawTriangle(const GSPrimitiveBatch &batch, const RasterContext &raster)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const GSVertex &v2 = batch.vertices[2];
    const auto &ctx = state.context;

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    float fx0 = v0.x - static_cast<float>(ofx);
    float fy0 = v0.y - static_cast<float>(ofy);
    float fx1 = v1.x - static_cast<float>(ofx);
    float fy1 = v1.y - static_cast<float>(ofy);
    float fx2 = v2.x - static_cast<float>(ofx);
    float fy2 = v2.y - static_cast<float>(ofy);

    int minX = static_cast<int>(std::floor(std::min({fx0, fx1, fx2})));
    int maxX = static_cast<int>(std::ceil(std::max({fx0, fx1, fx2})));
    int minY = static_cast<int>(std::floor(std::min({fy0, fy1, fy2})));
    int maxY = static_cast<int>(std::ceil(std::max({fy0, fy1, fy2})));

    minX = clampInt(minX, ctx.scissor.x0, ctx.scissor.x1);
    maxX = clampInt(maxX, ctx.scissor.x0, ctx.scissor.x1);
    minY = clampInt(minY, ctx.scissor.y0, ctx.scissor.y1);
    maxY = clampInt(maxY, ctx.scissor.y0, ctx.scissor.y1);

    float denom = (fy1 - fy2) * (fx0 - fx2) + (fx2 - fx1) * (fy0 - fy2);
    if (std::fabs(denom) < 0.001f)
        return;

    const float winding = (denom < 0.0f) ? -1.0f : 1.0f;
    const float invAbsDenom = 1.0f / std::fabs(denom);
    constexpr float kEdgeEpsilon = 1.0e-4f;

    // Each weight is linear in px: w = slope * (px - fx2) + rowTerm. Solving the three edge
    // conditions gives the row's covered span; scanning only that span (padded, the
    // per-pixel test below still decides) skips the empty part of the bounding box.
    const float slope0 = ((fy1 - fy2) * winding) * invAbsDenom;
    const float slope1 = ((fy2 - fy0) * winding) * invAbsDenom;

    for (int y = minY; y <= maxY; ++y)
    {
        float py = static_cast<float>(y) + 0.5f;

        const float row0 = (((fx2 - fx1) * (py - fy2)) * winding) * invAbsDenom;
        const float row1 = (((fx0 - fx2) * (py - fy2)) * winding) * invAbsDenom;
        float lo = static_cast<float>(minX) - 2.0f - fx2;
        float hi = static_cast<float>(maxX) + 2.0f - fx2;
        bool empty = false;
        auto bound = [&](float slope, float rowTerm)
        {
            // slope * d + rowTerm >= -kEdgeEpsilon
            if (slope > 0.0f)
                lo = std::max(lo, (-kEdgeEpsilon - rowTerm) / slope);
            else if (slope < 0.0f)
                hi = std::min(hi, (-kEdgeEpsilon - rowTerm) / slope);
            else if (rowTerm < -kEdgeEpsilon)
                empty = true;
        };
        bound(slope0, row0);
        bound(slope1, row1);
        bound(-(slope0 + slope1), 1.0f - row0 - row1);
        if (empty || lo > hi + 1.0f)
            continue;
        const int spanX0 = std::max(minX, floorToInt(lo + fx2 - 0.5f) - 1);
        const int spanX1 = std::min(maxX, floorToInt(hi + fx2 - 0.5f) + 2);

        for (int runStart = spanX0, runEnd = 0; raster.ownedRun(y, runStart, runEnd, spanX1); runStart = runEnd + 1)
        for (int x = runStart; x <= runEnd; ++x)
        {
            float px = static_cast<float>(x) + 0.5f;

            float w0 = (((fy1 - fy2) * (px - fx2) + (fx2 - fx1) * (py - fy2)) * winding) * invAbsDenom;
            float w1 = (((fy2 - fy0) * (px - fx2) + (fx0 - fx2) * (py - fy2)) * winding) * invAbsDenom;
            float w2 = 1.0f - w0 - w1;

            if (w0 < -kEdgeEpsilon || w1 < -kEdgeEpsilon || w2 < -kEdgeEpsilon)
                continue;

            double z = v0.z * w0 + v1.z * w1 + v2.z * w2;

            uint8_t r, g, b, a;
            if (state.prim.iip)
            {
                r = clampU8(static_cast<int>(v0.r * w0 + v1.r * w1 + v2.r * w2));
                g = clampU8(static_cast<int>(v0.g * w0 + v1.g * w1 + v2.g * w2));
                b = clampU8(static_cast<int>(v0.b * w0 + v1.b * w1 + v2.b * w2));
                a = clampU8(static_cast<int>(v0.a * w0 + v1.a * w1 + v2.a * w2));
            }
            else
            {
                r = v2.r;
                g = v2.g;
                b = v2.b;
                a = v2.a;
            }

            if (state.prim.tme)
            {
                float is, it, iq;
                uint16_t iu, iv;
                if (state.prim.fst)
                {
                    iu = static_cast<uint16_t>(v0.u * w0 + v1.u * w1 + v2.u * w2);
                    iv = static_cast<uint16_t>(v0.v * w0 + v1.v * w1 + v2.v * w2);
                    is = 0.0f;
                    it = 0.0f;
                    iq = 1.0f;
                }
                else
                {
                    // The GS DDA interpolates the homogeneous S, T and Q
                    // values. Texel coordinates are calculated from S/Q and
                    // T/Q only after interpolation.
                    is = v0.s * w0 + v1.s * w1 + v2.s * w2;
                    it = v0.t * w0 + v1.t * w1 + v2.t * w2;
                    iq = v0.q * w0 + v1.q * w1 + v2.q * w2;
                    iu = 0;
                    iv = 0;
                }

                uint32_t texel = SampleTexture(*raster.texture, *raster.cache, is, it, iq, iu, iv);

                uint8_t tr = static_cast<uint8_t>(texel & 0xFF);
                uint8_t tg = static_cast<uint8_t>((texel >> 8) & 0xFF);
                uint8_t tb = static_cast<uint8_t>((texel >> 16) & 0xFF);
                uint8_t ta = static_cast<uint8_t>((texel >> 24) & 0xFF);

                const auto &tex = ctx.tex0;
                const uint8_t shadeR = r;
                const uint8_t shadeG = g;
                const uint8_t shadeB = b;
                const uint8_t shadeA = a;
                const TextureCombineResult color = combineTexture(tex, shadeR, shadeG, shadeB, shadeA, tr, tg, tb, ta);

                r = color.r;
                g = color.g;
                b = color.b;
                a = color.a;
            }

            const uint8_t fog = clampU8(static_cast<int>(v0.fog * w0 + v1.fog * w1 + v2.fog * w2));
            WritePixel(*raster.pixel, x, y, static_cast<u32>(z + 0.5), r, g, b, a, fog);
        }
    }
}

void GSCpuBackend::DrawLine(const GSPrimitiveBatch &batch, const RasterContext &raster)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const auto &ctx = state.context;

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    int x0 = static_cast<int>(v0.x) - ofx;
    int y0 = static_cast<int>(v0.y) - ofy;
    int x1 = static_cast<int>(v1.x) - ofx;
    int y1 = static_cast<int>(v1.y) - ofy;

    int dx = std::abs(x1 - x0);
    int dy = -std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;

    int totalSteps = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
    if (totalSteps == 0)
        totalSteps = 1;
    int step = 0;

    for (;;)
    {
        float t = static_cast<float>(step) / static_cast<float>(totalSteps);
        uint8_t r, g, b, a;
        if (state.prim.iip)
        {
            r = clampU8(static_cast<int>(v0.r + (v1.r - v0.r) * t));
            g = clampU8(static_cast<int>(v0.g + (v1.g - v0.g) * t));
            b = clampU8(static_cast<int>(v0.b + (v1.b - v0.b) * t));
            a = clampU8(static_cast<int>(v0.a + (v1.a - v0.a) * t));
        }
        else
        {
            r = v1.r;
            g = v1.g;
            b = v1.b;
            a = v1.a;
        }

        double z = (v0.z + (v1.z - v0.z) * t);
        const uint8_t fog = clampU8(static_cast<int>(v0.fog + (v1.fog - v0.fog) * t));
        if (raster.owns(x0, y0))
            WritePixel(*raster.pixel, x0, y0, static_cast<u32>(z), r, g, b, a, fog);

        if (x0 == x1 && y0 == y1)
            break;

        int e2 = 2 * err;
        if (e2 >= dy)
        {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx)
        {
            err += dx;
            y0 += sy;
        }
        ++step;
    }
}

void GSCpuBackend::BeginTransfer(const GSTransferCommand &command)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (command.direction != 0u)
        DrainUnlocked();
    m_transfer = command;
    m_transferState.x = command.trxpos.dsax;
    m_transferState.y = command.trxpos.dsay;
    m_transferState.totalPixels = static_cast<uint32_t>(command.trxreg.rrw) * static_cast<uint32_t>(command.trxreg.rrh);
    m_transferState.copiedPixels = 0u;
    m_transferState.direction = command.direction;
    m_transferState.localToHostPendingBytes = 0u;

    if (command.direction == 2u)
        PerformLocalToLocalTransfer();
    else if (command.direction == 1u)
        PerformLocalToHostTransfer();
}

void GSCpuBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!data || sizeBytes == 0u || !m_vram || m_transferState.direction != 0u)
        return;
    if (m_transfer.trxreg.rrw == 0u || m_transfer.trxreg.rrh == 0u || m_transferState.totalPixels == 0u)
        return;

    const uint32_t firstPixel = m_transferState.copiedPixels;
    const uint32_t totalPixels = m_transferState.totalPixels;
    uint32_t consumed = 0u;
    if (m_dumpRemaining != 0u)
        DumpLog("UPLOAD dbp=%u dbw=%u dpsm=0x%x pos=%u,%u size=%ux%u first=%u bytes=%u", unsigned(m_transfer.bitbltbuf.dbp),
                unsigned(m_transfer.bitbltbuf.dbw), unsigned(m_transfer.bitbltbuf.dpsm), unsigned(m_transfer.trxpos.dsax),
                unsigned(m_transfer.trxpos.dsay), unsigned(m_transfer.trxreg.rrw), unsigned(m_transfer.trxreg.rrh), firstPixel, sizeBytes);
    bool direct = m_workerCount == 0u || m_dumpRemaining != 0u;
    consumed = WriteUploadPixels(m_transfer, firstPixel, totalPixels, data, sizeBytes, nullptr, direct);
    const uint32_t rrw = m_transfer.trxreg.rrw;
    const int firstRow = static_cast<int>(m_transfer.trxpos.dsay + firstPixel / rrw);
    const int lastRow = static_cast<int>(m_transfer.trxpos.dsay + (firstPixel + std::max(consumed, 1u) - 1u) / rrw);
    const auto target = rowBlocks(m_transfer.bitbltbuf.dbp, m_transfer.bitbltbuf.dbw, m_transfer.bitbltbuf.dpsm,
                                  firstRow, std::min(lastRow, 2047));
    if (!direct && !OverlapsInFlightUnlocked(target, true))
    {
        WriteUploadPixels(m_transfer, firstPixel, totalPixels, data, sizeBytes, nullptr, true);
        m_callerCacheStale = true;
        NoteBarrierRange(m_pixelWritesSinceBarrier, target);
        direct = true;
    }
    if (direct)
    {
        NoteDirectWrite(target);
    }
    else
    {
        // Queued like a primitive: each worker writes the destination rows it owns, so pixel
        // order with drawing is kept. Queued primitives reading the destination, or writing
        // depth there, must finish first (barrier); later ones reading it get a barrier through
        // the write ranges.
        const bool barrier = m_barrierRequested || overlapsAny(target, m_readsSinceBarrier) ||
                             overlapsAny(target, m_zWritesSinceBarrier);
        if (barrier)
        {
            m_pixelWritesSinceBarrier.clear();
            m_zWritesSinceBarrier.clear();
            m_readsSinceBarrier.clear();
            m_zOwnerKeys.clear();
            m_barrierRequested = false;
        }
        QueuedPrimitive &item = ReserveQueueSlotUnlocked();
        item.textureFlush = false;
        item.upload = true;
        item.barrier = barrier;
        item.area = 0;
        item.transfer = m_transfer;
        item.uploadFirstPixel = firstPixel;
        item.uploadTotalPixels = totalPixels;
        item.uploadData.assign(data, data + sizeBytes);
        item.writes[0] = target;
        item.writeCount = 1u;
        item.samples = false;
        NotePendingRange(m_pendingWrites, target);
        NoteBarrierRange(m_pixelWritesSinceBarrier, target);
        PublishQueueSlotUnlocked();
    }

    m_transferState.copiedPixels = std::min<uint32_t>(totalPixels, firstPixel + consumed);
    if (m_transferState.copiedPixels >= totalPixels)
    {
        m_transferState.direction = 3u;
        m_transferState.totalPixels = 0u;
    }
    else
    {
        m_transferState.x = m_transfer.trxpos.dsax + (m_transferState.copiedPixels % m_transfer.trxreg.rrw);
        m_transferState.y = m_transfer.trxpos.dsay + (m_transferState.copiedPixels / m_transfer.trxreg.rrw);
    }
}

uint32_t GSCpuBackend::WriteUploadPixels(const GSTransferCommand &transfer,
                                         uint32_t firstPixel,
                                         uint32_t totalPixels,
                                         const uint8_t *data,
                                         uint32_t sizeBytes,
                                         const RasterContext *rows,
                                         bool write)
{
    const uint32_t dbp = transfer.bitbltbuf.dbp;
    const uint32_t dbw = std::max<uint32_t>(transfer.bitbltbuf.dbw, 1u);
    const uint8_t dpsm = transfer.bitbltbuf.dpsm;
    const uint32_t rrw = transfer.trxreg.rrw;
    const uint32_t dsax = transfer.trxpos.dsax;
    const uint32_t dsay = transfer.trxpos.dsay;
    uint32_t pixel = firstPixel;
    uint32_t offset = 0u;

    auto put = [&](uint32_t index, uint32_t value)
    {
        const uint32_t x = dsax + index % rrw;
        const uint32_t y = dsay + index / rrw;
        if (write && (!rows || (GSMem::PageOf(dpsm, dbp, dbw, x, y) % rows->stripeCount) == rows->stripe))
            WriteVramUnlocked(dpsm, dbp, dbw, x, y, value);
    };

    while (offset < sizeBytes && pixel < totalPixels)
    {
        switch (dpsm)
        {
        case GS_PSM_CT32:
        case GS_PSM_Z32:
        {
            if (sizeBytes - offset < 4u)
                return pixel - firstPixel;
            uint32_t value = 0u;
            std::memcpy(&value, data + offset, sizeof(value));
            put(pixel, value);
            offset += 4u;
            ++pixel;
            break;
        }
        case GS_PSM_CT24:
        case GS_PSM_Z24:
        {
            if (sizeBytes - offset < 3u)
                return pixel - firstPixel;
            const uint32_t value = static_cast<uint32_t>(data[offset]) |
                                   (static_cast<uint32_t>(data[offset + 1u]) << 8u) |
                                   (static_cast<uint32_t>(data[offset + 2u]) << 16u);
            put(pixel, value);
            offset += 3u;
            ++pixel;
            break;
        }
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
        {
            if (sizeBytes - offset < 2u)
                return pixel - firstPixel;
            uint16_t value = 0u;
            std::memcpy(&value, data + offset, sizeof(value));
            put(pixel, value);
            offset += 2u;
            ++pixel;
            break;
        }
        case GS_PSM_T8:
        case GS_PSM_T8H:
            put(pixel, data[offset++]);
            ++pixel;
            break;
        case GS_PSM_T4:
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
        {
            const uint8_t packed = data[offset++];
            put(pixel, packed & 0x0Fu);
            if (pixel + 1u < totalPixels)
                put(pixel + 1u, (packed >> 4u) & 0x0Fu);
            pixel += std::min<uint32_t>(2u, totalPixels - pixel);
            break;
        }
        default:
            return pixel - firstPixel;
        }
    }
    return pixel - firstPixel;
}

void GSCpuBackend::PerformLocalToLocalTransfer()
{
    if (!m_vram)
        return;

    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    const uint32_t total = rrw * rrh;
    if (total == 0u)
    {
        m_transferState.direction = 3u;
        return;
    }

    NoteDirectWrite(rowBlocks(m_transfer.bitbltbuf.dbp, m_transfer.bitbltbuf.dbw, m_transfer.bitbltbuf.dpsm,
                              static_cast<int>(m_transfer.trxpos.dsay), static_cast<int>(m_transfer.trxpos.dsay + rrh - 1u)));
    for (uint32_t pixel = 0; pixel < total; ++pixel)
    {
        uint32_t x = pixel % rrw;
        uint32_t y = pixel / rrw;
        if ((m_transfer.trxpos.dir & 0x2u) != 0u)
            x = rrw - x - 1u;
        if ((m_transfer.trxpos.dir & 0x1u) != 0u)
            y = rrh - y - 1u;

        const uint32_t value = ReadVramUnlocked(m_transfer.bitbltbuf.spsm,
                                                m_transfer.bitbltbuf.sbp,
                                                std::max<uint32_t>(m_transfer.bitbltbuf.sbw, 1u),
                                                x + m_transfer.trxpos.ssax,
                                                y + m_transfer.trxpos.ssay);
        WriteVramUnlocked(m_transfer.bitbltbuf.dpsm,
                          m_transfer.bitbltbuf.dbp,
                          std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u),
                          x + m_transfer.trxpos.dsax,
                          y + m_transfer.trxpos.dsay,
                          value);
    }

    m_transferState.copiedPixels = total;
    m_transferState.direction = 3u;
}

void GSCpuBackend::PerformLocalToHostTransfer()
{
    m_localToHostBuffer.clear();
    m_localToHostReadPos = 0u;
    if (!m_vram)
        return;

    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    const uint32_t sbw = std::max<uint32_t>(m_transfer.bitbltbuf.sbw, 1u);
    const uint8_t spsm = m_transfer.bitbltbuf.spsm;
    const uint32_t bpp = static_cast<uint32_t>(GSMem::BitsPerPixel(static_cast<GSMem::PixelStorageMode>(spsm)));
    const uint32_t total = rrw * rrh;
    m_localToHostBuffer.reserve((static_cast<size_t>(total) * bpp + 7u) / 8u);

    for (uint32_t pixel = 0u; pixel < total; ++pixel)
    {
        const uint32_t x = pixel % rrw;
        const uint32_t y = pixel / rrw;
        const uint32_t value = ReadVramUnlocked(spsm,
                                                m_transfer.bitbltbuf.sbp,
                                                sbw,
                                                x + m_transfer.trxpos.ssax,
                                                y + m_transfer.trxpos.ssay);
        switch (bpp)
        {
        case 32:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 16u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 24u));
            break;
        case 24:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 16u));
            break;
        case 16:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            break;
        case 8:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            break;
        case 4:
        {
            if ((pixel & 1u) != 0u)
                break;
            uint32_t next = 0u;
            if (pixel + 1u < total)
            {
                const uint32_t nextPixel = pixel + 1u;
                const uint32_t nextX = nextPixel % rrw;
                const uint32_t nextY = nextPixel / rrw;
                next = ReadVramUnlocked(spsm, m_transfer.bitbltbuf.sbp, sbw,
                                        nextX + m_transfer.trxpos.ssax,
                                        nextY + m_transfer.trxpos.ssay);
            }
            m_localToHostBuffer.push_back(static_cast<uint8_t>((value & 0x0Fu) | ((next & 0x0Fu) << 4u)));
            break;
        }
        default:
            break;
        }
    }

    m_transferState.copiedPixels = total;
    m_transferState.localToHostPendingBytes = m_localToHostBuffer.size();
}

uint32_t GSCpuBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    DrainUnlocked();
    if (!dst || maxBytes == 0u || m_localToHostReadPos >= m_localToHostBuffer.size())
        return 0u;
    const size_t count = std::min<size_t>(maxBytes, m_localToHostBuffer.size() - m_localToHostReadPos);
    std::memcpy(dst, m_localToHostBuffer.data() + m_localToHostReadPos, count);
    m_localToHostReadPos += count;
    m_transferState.localToHostPendingBytes = m_localToHostBuffer.size() - m_localToHostReadPos;
    return static_cast<uint32_t>(count);
}

bool GSCpuBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    DrainUnlocked();
    if (!m_vram || context.frame.fbw == 0u)
        return false;

    const uint32_t x0 = context.scissor.x0;
    const uint32_t x1 = std::max<uint32_t>(x0, context.scissor.x1);
    const uint32_t y0 = context.scissor.y0;
    const uint32_t y1 = std::max<uint32_t>(y0, context.scissor.y1);
    uint8_t r = static_cast<uint8_t>(rgba);
    uint8_t g = static_cast<uint8_t>(rgba >> 8u);
    uint8_t b = static_cast<uint8_t>(rgba >> 16u);
    uint8_t a = static_cast<uint8_t>(rgba >> 24u);
    if ((context.fba & 1ull) != 0ull && context.frame.psm != GS_PSM_CT24)
        a |= 0x80u;

    const uint32_t fbp = GSInternal::framePageBaseToBlock(context.frame.fbp);
    const uint32_t fbw = std::max<uint32_t>(context.frame.fbw, 1u);
    NoteDirectWrite(rowBlocks(fbp, fbw, context.frame.psm, static_cast<int>(y0), static_cast<int>(y1)));
    if (context.frame.psm == GS_PSM_CT32 || context.frame.psm == GS_PSM_CT24)
    {
        const uint32_t source = static_cast<uint32_t>(r) |
                                (static_cast<uint32_t>(g) << 8u) |
                                (static_cast<uint32_t>(b) << 16u) |
                                (static_cast<uint32_t>(a) << 24u);
        for (uint32_t y = y0; y <= y1; ++y)
            for (uint32_t x = x0; x <= x1; ++x)
            {
                uint32_t pixel = source;
                if (context.frame.fbmsk != 0u)
                {
                    const uint32_t old = ReadVramUnlocked(context.frame.psm, fbp, fbw, x, y);
                    pixel = (pixel & ~context.frame.fbmsk) | (old & context.frame.fbmsk);
                }
                WriteVramUnlocked(context.frame.psm, fbp, fbw, x, y, pixel);
            }
        return true;
    }

    if (context.frame.psm == GS_PSM_CT16 || context.frame.psm == GS_PSM_CT16S)
    {
        const uint16_t source = encodeFramePixelPSMCT16(r, g, b, a);
        const uint16_t mask = static_cast<uint16_t>(context.frame.fbmsk);
        for (uint32_t y = y0; y <= y1; ++y)
            for (uint32_t x = x0; x <= x1; ++x)
            {
                uint16_t pixel = source;
                if (mask != 0u)
                {
                    const uint16_t old = static_cast<uint16_t>(ReadVramUnlocked(context.frame.psm, fbp, fbw, x, y));
                    pixel = static_cast<uint16_t>((pixel & ~mask) | (old & mask));
                }
                WriteVramUnlocked(context.frame.psm, fbp, fbw, x, y, pixel);
            }
        return true;
    }
    return false;
}

bool GSCpuBackend::CopyFrameToHostRgba(const GSFrameReg &frame,
                                       uint32_t width,
                                       uint32_t height,
                                       std::vector<uint8_t> &outPixels,
                                       bool preserveAlpha,
                                       bool useLocalMemoryLayout,
                                       bool frameBaseIsPages,
                                       uint32_t sourceOriginX,
                                       uint32_t sourceOriginY) const
{
    if (!m_vram || m_vramSize == 0u)
        return false;

    outPixels.assign(kHostFrameWidth * kHostFrameHeight * 4u, 0u);
    const uint32_t baseBytes = frameBaseIsPages ? frame.fbp * 8192u : frame.fbp * 256u;
    const uint32_t basePtr = frameBaseIsPages ? GSInternal::framePageBaseToBlock(frame.fbp) : frame.fbp;
    const uint32_t fbw = frame.fbw ? frame.fbw : kHostFrameWidth / 64u;
    const uint32_t bytesPerPixel = (frame.psm == GS_PSM_CT16 || frame.psm == GS_PSM_CT16S) ? 2u : 4u;
    const uint32_t stride = fbw * 64u * bytesPerPixel;

    for (uint32_t y = 0; y < height; ++y)
    {
        uint8_t *dst = outPixels.data() + y * kHostFrameWidth * 4u;
        for (uint32_t x = 0; x < width; ++x)
        {
            const uint32_t sx = sourceOriginX + x;
            const uint32_t sy = sourceOriginY + y;
            if (frame.psm == GS_PSM_CT32 || frame.psm == GS_PSM_CT24)
            {
                uint32_t color = 0u;
                if (useLocalMemoryLayout)
                    color = ReadVramUnlocked(frame.psm, basePtr, fbw, sx, sy);
                else
                {
                    const uint32_t pixelBytes = frame.psm == GS_PSM_CT24 ? 3u : 4u;
                    const uint64_t offset = static_cast<uint64_t>(baseBytes) + static_cast<uint64_t>(sy) * stride + static_cast<uint64_t>(sx) * pixelBytes;
                    if (offset + pixelBytes > m_vramSize)
                        return false;
                    color = m_vram[offset] | (static_cast<uint32_t>(m_vram[offset + 1u]) << 8u) |
                            (static_cast<uint32_t>(m_vram[offset + 2u]) << 16u);
                    if (pixelBytes == 4u)
                        color |= static_cast<uint32_t>(m_vram[offset + 3u]) << 24u;
                }
                dst[x * 4u] = static_cast<uint8_t>(color);
                dst[x * 4u + 1u] = static_cast<uint8_t>(color >> 8u);
                dst[x * 4u + 2u] = static_cast<uint8_t>(color >> 16u);
                dst[x * 4u + 3u] = preserveAlpha && frame.psm != GS_PSM_CT24 ? static_cast<uint8_t>(color >> 24u) : 255u;
            }
            else if (frame.psm == GS_PSM_CT16 || frame.psm == GS_PSM_CT16S)
            {
                uint16_t color = 0u;
                if (useLocalMemoryLayout)
                    color = static_cast<uint16_t>(ReadVramUnlocked(frame.psm, basePtr, fbw, sx, sy));
                else
                {
                    const uint64_t offset = static_cast<uint64_t>(baseBytes) + static_cast<uint64_t>(sy) * stride + static_cast<uint64_t>(sx) * 2u;
                    if (offset + 2u > m_vramSize)
                        return false;
                    std::memcpy(&color, m_vram + offset, sizeof(color));
                }
                const uint32_t r = color & 31u;
                const uint32_t g = (color >> 5u) & 31u;
                const uint32_t b = (color >> 10u) & 31u;
                dst[x * 4u] = static_cast<uint8_t>((r << 3u) | (r >> 2u));
                dst[x * 4u + 1u] = static_cast<uint8_t>((g << 3u) | (g >> 2u));
                dst[x * 4u + 2u] = static_cast<uint8_t>((b << 3u) | (b >> 2u));
                dst[x * 4u + 3u] = preserveAlpha ? ((color & 0x8000u) ? 0x80u : 0u) : 255u;
            }
            else
            {
                outPixels.clear();
                return false;
            }
        }
    }
    return true;
}

PresentationFrame GSCpuBackend::Present(const GSPresentationRequest &request)
{
    // Snapshot local memory under the backend lock, then perform the expensive
    // display conversion without holding the producer-side raster lock.
    thread_local std::vector<uint8_t> snapshot;
    SnapshotVram(snapshot);
    if (snapshot.empty())
        return {};

    thread_local GSCpuBackend snapshotBackend;
    snapshotBackend.Initialize(snapshot.data(), static_cast<uint32_t>(snapshot.size()));
    return snapshotBackend.PresentFromLocalMemory(request);
}

PresentationFrame GSCpuBackend::PresentFromLocalMemory(const GSPresentationRequest &request)
{
    PresentationFrame result{};
    const GSPmodeState pmode = decodePmode(request.pmode);
    const GSSmode2State smode2 = decodeSMode2(request.smode2);
    const bool fieldMode = smode2.interlaced && !smode2.frameMode;
    const bool oddField = (request.vsyncTick & 1ull) != 0ull;
    const GSFrameReg displayFrame1 = decodeDisplayFrame(request.dispfb1);
    const GSFrameReg displayFrame2 = decodeDisplayFrame(request.dispfb2);
    const GSDisplayReadOrigin origin1 = decodeDisplayReadOrigin(request.dispfb1);
    const GSDisplayReadOrigin origin2 = decodeDisplayReadOrigin(request.dispfb2);
    uint32_t width1 = 0u, height1 = 0u, width2 = 0u, height2 = 0u;
    decodeDisplaySize(request.display1, width1, height1);
    decodeDisplaySize(request.display2, width2, height2);
    const bool frameModeInterlace = smode2.interlaced && smode2.frameMode;
    const uint32_t displayHeight1 = height1;
    const uint32_t displayHeight2 = height2;
    if (frameModeInterlace)
    {
        height1 = (height1 + 1u) / 2u;
        height2 = (height2 + 1u) / 2u;
    }
    const bool valid1 = pmode.enableCrt1 && hasDisplaySetup(request.display1, displayFrame1);
    const bool valid2 = pmode.enableCrt2 && hasDisplaySetup(request.display2, displayFrame2);
    if (!valid1 && !valid2)
        return result;

    auto copySource = [&](const GSFrameReg &displayFrame,
                          const GSDisplayReadOrigin &origin,
                          uint32_t width,
                          uint32_t height,
                          bool allowPreferred,
                          bool preserveAlpha,
                          GSFrameReg &selected,
                          std::vector<uint8_t> &pixels,
                          bool &usedPreferred) -> bool
    {
        selected = displayFrame;
        pixels.clear();
        usedPreferred = false;
        if (allowPreferred && request.hasPreferredSource && request.preferredDestFbp == displayFrame.fbp &&
            (request.preferredSource.fbw != 0u || request.preferredSource.fbp != displayFrame.fbp) &&
            CopyFrameToHostRgba(request.preferredSource, width, height, pixels, preserveAlpha, true, false, 0u, 0u))
        {
            selected = request.preferredSource;
            usedPreferred = true;
        }
        if (pixels.empty() && !CopyFrameToHostRgba(displayFrame, width, height, pixels, preserveAlpha, true, true, origin.x, origin.y))
            return false;

        if (!usedPreferred && displayFrame.fbp == 0u && countNonBlackPixels(pixels, width, height) == 0u)
        {
            for (const GSFrameReg &candidate : request.contextFrames)
            {
                if (candidate.fbp == selected.fbp && candidate.fbw == selected.fbw && candidate.psm == selected.psm)
                    continue;
                std::vector<uint8_t> candidatePixels;
                if (!CopyFrameToHostRgba(candidate, width, height, candidatePixels, preserveAlpha, true, true, 0u, 0u))
                    continue;
                if (countNonBlackPixels(candidatePixels, width, height) == 0u)
                    continue;
                selected = candidate;
                pixels.swap(candidatePixels);
                break;
            }
        }
        return true;
    };

    if (valid1 && valid2)
    {
        GSFrameReg selected1{}, selected2{};
        std::vector<uint8_t> crt1, crt2;
        bool preferred1 = false, preferred2 = false;
        if (copySource(displayFrame1, origin1, width1, height1, false, true, selected1, crt1, preferred1) &&
            copySource(displayFrame2, origin2, width2, height2, false, true, selected2, crt2, preferred2))
        {
            result.width = std::max(width1, width2);
            result.height = std::max(height1, height2);
            result.pixels.assign(kHostFrameWidth * kHostFrameHeight * 4u, 0u);
            const uint8_t bgR = static_cast<uint8_t>(request.bgcolor);
            const uint8_t bgG = static_cast<uint8_t>(request.bgcolor >> 8u);
            const uint8_t bgB = static_cast<uint8_t>(request.bgcolor >> 16u);
            for (uint32_t y = 0; y < result.height; ++y)
                for (uint32_t x = 0; x < result.width; ++x)
                {
                    uint8_t *dst = result.pixels.data() + (y * kHostFrameWidth + x) * 4u;
                    dst[0] = bgR;
                    dst[1] = bgG;
                    dst[2] = bgB;
                    dst[3] = pmode.alp;
                }
            if (!pmode.slbg)
                for (uint32_t y = 0; y < height2; ++y)
                    std::memcpy(result.pixels.data() + y * kHostFrameWidth * 4u, crt2.data() + y * kHostFrameWidth * 4u, width2 * 4u);
            for (uint32_t y = 0; y < height1; ++y)
                for (uint32_t x = 0; x < width1; ++x)
                {
                    const uint8_t *src = crt1.data() + (y * kHostFrameWidth + x) * 4u;
                    uint8_t *dst = result.pixels.data() + (y * kHostFrameWidth + x) * 4u;
                    const uint32_t factor = pmode.mmod ? pmode.alp : std::min<uint32_t>(255u, static_cast<uint32_t>(src[3]) * 2u);
                    dst[0] = blendPresentationChannel(src[0], dst[0], factor);
                    dst[1] = blendPresentationChannel(src[1], dst[1], factor);
                    dst[2] = blendPresentationChannel(src[2], dst[2], factor);
                    dst[3] = pmode.amod ? dst[3] : src[3];
                }
            normalizePresentationAlpha(result.pixels, result.width, result.height);
            if (fieldMode)
                applyFieldPresentation(result.pixels, result.width, result.height, oddField);
            if (frameModeInterlace)
            {
                const uint32_t displayHeight = std::max(displayHeight1, displayHeight2);
                applyFrameModeStabilize(result.pixels, result.width, result.height, displayHeight);
                result.height = displayHeight;
            }
            result.displayFbp = displayFrame1.fbp;
            result.sourceFbp = selected1.fbp;
            return result;
        }
    }

    const GSFrameReg &displayFrame = valid1 ? displayFrame1 : displayFrame2;
    const GSDisplayReadOrigin &origin = valid1 ? origin1 : origin2;
    result.width = valid1 ? width1 : width2;
    result.height = valid1 ? height1 : height2;
    GSFrameReg selected = displayFrame;
    if (!copySource(displayFrame, origin, result.width, result.height, true, false, selected, result.pixels, result.usedPreferred))
        return {};
    if (fieldMode)
        applyFieldPresentation(result.pixels, result.width, result.height, oddField);
    if (frameModeInterlace)
    {
        const uint32_t displayHeight = valid1 ? displayHeight1 : displayHeight2;
        applyFrameModeStabilize(result.pixels, result.width, result.height, displayHeight);
        result.height = displayHeight;
    }
    normalizePresentationAlpha(result.pixels, result.width, result.height);
    result.displayFbp = displayFrame.fbp;
    result.sourceFbp = selected.fbp;
    return result;
}

// ---------------------------------------------------------------- frame dump (diagnosis)

void GSCpuBackend::DumpUpdateUnlocked()
{
    if (m_dumpState == 2u)
        return;
    if (m_dumpState == 0u)
    {
        const char *dir = std::getenv("PS2X_GS_DUMP");
        if (!dir || !*dir)
        {
            m_dumpState = 2u;
            return;
        }
        m_dumpDir = dir;
        const char *after = std::getenv("PS2X_GS_DUMP_AFTER");
        m_dumpStart = std::chrono::steady_clock::now() + std::chrono::milliseconds(static_cast<int64_t>((after ? std::atof(after) : 0.0) * 1000.0));
        m_dumpState = 1u;
    }
    if (m_dumpRemaining != 0u || std::chrono::steady_clock::now() < m_dumpStart)
        return;
    // Start: finish queued work, then draw immediately (VRAM is exact at each primitive).
    DrainUnlocked();
    for (GSMem::TexturePageCache &cache : m_texturePageCaches)
        cache.Invalidate();
    for (DecodedTextureCache &cache : m_decodedCaches)
        cache.Invalidate();
    if (FILE *f = std::fopen((m_dumpDir + "/vram_before.bin").c_str(), "wb"))
    {
        std::fwrite(m_vram, 1, GSMem::MEMORY_SIZE, f);
        std::fclose(f);
    }
    m_dumpLog = std::fopen((m_dumpDir + "/prims.txt").c_str(), "w");
    const char *prims = std::getenv("PS2X_GS_DUMP_PRIMS");
    m_dumpTotal = prims && *prims ? std::max(1, std::atoi(prims)) : 4000u;
    m_dumpRemaining = m_dumpTotal;
    m_dumpState = 2u;
}

void GSCpuBackend::DumpLog(const char *format, ...) const
{
    if (!m_dumpLog)
        return;
    va_list args;
    va_start(args, format);
    std::vfprintf(m_dumpLog, format, args);
    va_end(args);
    std::fputc('\n', m_dumpLog);
}

void GSCpuBackend::DumpPrimitiveUnlocked(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const auto &ctx = state.context;
    const auto &tex = ctx.tex0;
    // During the second dumped frame, save the target after each group of primitives (same
    // target and texture), to see the frame being composed.
    const uint64_t group = (static_cast<uint64_t>(ctx.frame.fbp) << 40u) ^ (static_cast<uint64_t>(state.prim.tme ? tex.tbp0 + 1u : 0u) << 8u) ^
                           (static_cast<uint64_t>(ctx.alpha & 0xFFu)) ^ (static_cast<uint64_t>(state.prim.abe) << 63u);
    static const uint32_t snapPresents = []
    {
        const char *value = std::getenv("PS2X_GS_DUMP_SNAP_PRESENTS");
        return value && *value ? static_cast<uint32_t>(std::max(1, std::atoi(value))) : 1u;
    }();
    if (m_dumpPresents >= 1u && m_dumpPresents <= snapPresents && m_dumpGroupValid && group != m_dumpGroup)
    {
        const GSFrameReg &frame = m_dumpGroupFrame;
        const uint32_t width = std::min<uint32_t>(m_dumpGroupScissorX1 + 1u, 1024u);
        const uint32_t height = std::min<uint32_t>(m_dumpGroupScissorY1 + 1u, 1024u);
        std::vector<uint32_t> pixels(static_cast<size_t>(width) * height);
        const uint32_t base = GSInternal::framePageBaseToBlock(frame.fbp);
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x)
            {
                uint32_t value = ReadVramUnlocked(frame.psm, base, std::max<uint32_t>(frame.fbw, 1u), x, y);
                if (bitsPerPixel(frame.psm) == 16)
                    value = Rgba5551ToRgba8888(static_cast<u16>(value));
                else if (frame.psm == GS_PSM_CT24)
                    value |= 0x80000000u;
                pixels[static_cast<size_t>(y) * width + x] = value;
            }
        char name[96];
        std::snprintf(name, sizeof(name), "/snap_%04u_fbp%u_%ux%u.rgba", m_dumpSnapshots++, unsigned(frame.fbp), width, height);
        if (FILE *f = std::fopen((m_dumpDir + name).c_str(), "wb"))
        {
            std::fwrite(pixels.data(), 4, pixels.size(), f);
            std::fclose(f);
        }
        DumpLog("SNAP %s", name + 1);
    }
    m_dumpGroup = group;
    m_dumpGroupValid = true;
    m_dumpGroupFrame = ctx.frame;
    m_dumpGroupScissorX1 = ctx.scissor.x1;
    m_dumpGroupScissorY1 = ctx.scissor.y1;
    const uint32_t count = std::min<uint32_t>(primitiveVertexCount(state.prim.type), batch.vertexCount);
    char vertices[512];
    int length = 0;
    for (uint32_t i = 0; i < count; ++i)
    {
        const GSVertex &v = batch.vertices[i];
        length += std::snprintf(vertices + length, sizeof(vertices) - length, " v%u=(%.1f,%.1f z=%.0f rgba=%02x%02x%02x%02x uv=%u,%u stq=%.4f,%.4f,%.4f)", i,
                                v.x - (ctx.xyoffset.ofx >> 4), v.y - (ctx.xyoffset.ofy >> 4), v.z, v.r, v.g, v.b, v.a, v.u >> 4, v.v >> 4, v.s, v.t, v.q);
    }
    int textureIndex = -1;
    if (state.prim.tme)
    {
        // Decode each distinct texture once, as the sampler sees it (CLUT and TEXA applied).
        TextureSampler sampler = MakeTextureSampler(state);
        const auto key = std::make_tuple(tex.tbp0, tex.tbw, tex.psm, sampler.texW, sampler.texH, tex.cpsm, tex.csm, tex.csa, m_clutSerial, state.texa.ta0, state.texa.ta1, state.texa.aem);
        auto it = m_dumpTextures.find(key);
        if (it == m_dumpTextures.end())
        {
            const int index = static_cast<int>(m_dumpTextures.size());
            it = m_dumpTextures.emplace(key, index).first;
            std::vector<uint32_t> texels(static_cast<size_t>(sampler.texW) * sampler.texH);
            GSMem::TexturePageCache fresh;
            for (int y = 0; y < sampler.texH; ++y)
                for (int x = 0; x < sampler.texW; ++x)
                    texels[static_cast<size_t>(y) * sampler.texW + x] = FetchTexel(sampler, fresh, x, y);
            char name[64];
            std::snprintf(name, sizeof(name), "/tex_%04d_%dx%d.rgba", index, sampler.texW, sampler.texH);
            if (FILE *f = std::fopen((m_dumpDir + name).c_str(), "wb"))
            {
                std::fwrite(texels.data(), 4, texels.size(), f);
                std::fclose(f);
            }
        }
        textureIndex = it->second;
    }
    DumpLog("PRIM #%u type=%u iip=%u tme=%u fge=%u abe=%u fst=%u ctxt=%u frame=(fbp=%u fbw=%u psm=0x%x msk=%08x) zbuf=(zbp=%u psm=0x%x zmsk=%u) "
            "scissor=%u,%u-%u,%u test=0x%llx alpha=0x%llx fba=%u pabe=%u tex%d=(tbp0=%u tbw=%u psm=0x%x tw=%u th=%u tcc=%u tfx=%u cbp=%u cpsm=0x%x csm=%u csa=%u) "
            "lin=%u clamp=0x%llx texa=%u/%u/%u%s",
            m_dumpTotal - m_dumpRemaining, unsigned(state.prim.type), unsigned(state.prim.iip), unsigned(state.prim.tme), unsigned(state.prim.fge), unsigned(state.prim.abe),
            unsigned(state.prim.fst), unsigned(state.prim.ctxt), unsigned(ctx.frame.fbp), unsigned(ctx.frame.fbw), unsigned(ctx.frame.psm), unsigned(ctx.frame.fbmsk),
            unsigned(ctx.zbuf.zbp), unsigned(ctx.zbuf.psm), unsigned(ctx.zbuf.zmask), unsigned(ctx.scissor.x0), unsigned(ctx.scissor.y0), unsigned(ctx.scissor.x1),
            unsigned(ctx.scissor.y1), (unsigned long long)ctx.test, (unsigned long long)ctx.alpha, unsigned(ctx.fba & 1u), unsigned(state.pabe), textureIndex,
            unsigned(tex.tbp0), unsigned(tex.tbw), unsigned(tex.psm), unsigned(tex.tw), unsigned(tex.th), unsigned(tex.tcc), unsigned(tex.tfx), unsigned(tex.cbp),
            unsigned(tex.cpsm), unsigned(tex.csm), unsigned(tex.csa), unsigned(state.linearFilter), (unsigned long long)ctx.clamp, unsigned(state.texa.ta0),
            unsigned(state.texa.ta1), unsigned(state.texa.aem), vertices);
}

void GSCpuBackend::DumpFinishUnlocked()
{
    if (FILE *f = std::fopen((m_dumpDir + "/vram_after.bin").c_str(), "wb"))
    {
        std::fwrite(m_vram, 1, GSMem::MEMORY_SIZE, f);
        std::fclose(f);
    }
    if (m_dumpLog)
    {
        std::fclose(m_dumpLog);
        m_dumpLog = nullptr;
    }
    // Workers were idle meanwhile; their caches predate the dumped drawing.
    for (GSMem::TexturePageCache &cache : m_texturePageCaches)
        cache.Invalidate();
    for (DecodedTextureCache &cache : m_decodedCaches)
        cache.Invalidate();
}
