#pragma once

#include "runtime/gs/gs_backend.h"
#include "runtime/gs/gs_texture_page_cache.h"

#include <array>
#include <functional>
#include <mutex>
#include <vector>

class GSCpuBackend final : public GSRasterBackend
{
public:
    GSCpuBackend();

    void Initialize(uint8_t *vram, uint32_t vramSize) override;
    void Reset() override;

    void Submit(const GSPrimitiveBatch &batch) override;
    void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override;
    void BeginTransfer(const GSTransferCommand &command) override;
    void UploadImage(const uint8_t *data, uint32_t sizeBytes) override;

    void Flush() override;
    void TextureFlush() override;
    void Sync(GSSyncReason reason) override;
    PresentationFrame Present(const GSPresentationRequest &request) override;

    bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override;
    uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override;

    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override;
    void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override;
    void SnapshotVram(std::vector<uint8_t> &out) const override;
    GSTransferSnapshot GetTransferSnapshot() const override;

private:
    void ResetUnlocked();
    void LoadClutUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut);
    uint32_t ReadVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const;
    uint32_t ReadTextureVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y);
    void WriteVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value);

    void DrawPrimitive(const GSPrimitiveBatch &batch);
    void DrawSprite(const GSPrimitiveBatch &batch);
    void DrawTriangle(const GSPrimitiveBatch &batch);
    void DrawLine(const GSPrimitiveBatch &batch);
    void WritePixel(const GSDrawState &state, int x, int y, int z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog);
    uint32_t SampleTexture(const GSDrawState &state, float s, float t, float q, uint16_t u, uint16_t v);
    uint32_t LookupCLUT(const GSDrawState &state, uint8_t index, uint8_t cpsm, uint8_t csm, uint8_t csa, uint8_t sourcePsm,
                        const uint16_t *clut);

    void PerformLocalToLocalTransfer();
    void PerformLocalToHostTransfer();
    PresentationFrame PresentFromLocalMemory(const GSPresentationRequest &request);
    bool CopyFrameToHostRgba(const GSFrameReg &frame,
                             uint32_t width,
                             uint32_t height,
                             std::vector<uint8_t> &outPixels,
                             bool preserveAlpha,
                             bool useLocalMemoryLayout,
                             bool frameBaseIsPages,
                             uint32_t sourceOriginX,
                             uint32_t sourceOriginY) const;

    using WriteVramFunc = void (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    using ReadVramFunc = uint32_t (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t);

    static constexpr size_t kPsmHandlerCount = 1u << 6u;
    mutable std::mutex m_mutex;
    uint8_t *m_vram = nullptr;
    uint32_t m_vramSize = 0;
    std::array<ReadVramFunc, kPsmHandlerCount> m_readVramFuncs{};
    std::array<WriteVramFunc, kPsmHandlerCount> m_writeVramFuncs{};
    std::array<uint16_t, 512> m_clut{};
    std::array<uint32_t, 2> m_clutCbp{};
    GSMem::TexturePageCache m_texturePageCache;

    // Lazily decoded RGBA texels of the texture bound to the primitive being drawn. A texel is valid
    // when its stamp equals the epoch; the epoch advances when the texture identity, its VRAM pages
    // (GSMem::g_pageWriteGeneration) or the CLUT buffer change.
    struct TexelCacheKey
    {
        uint32_t tbp0 = 0;
        uint32_t tbw = 0;
        uint32_t psm = 0;
        uint32_t cpsm = 0;
        uint32_t csm = 0;
        uint32_t csa = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t texa = 0;
        uint32_t clutGeneration = 0;
        bool operator==(const TexelCacheKey &) const = default;
    };
    // A decoded-texel cache for one texture, with the CLUT as it was when the slot was prepared (queued
    // primitives keep sampling with it after later CLUT loads).
    struct TexelCacheSlot
    {
        TexelCacheKey key{};
        std::vector<std::pair<uint32_t, uint32_t>> pages; // (page, write generation)
        std::array<uint16_t, 512> clut{};
        std::vector<uint32_t> texels;
        std::vector<uint32_t> stamps;
        uint32_t epoch = 1u;
        uint64_t lastUse = 0u;
        bool valid = false;
        bool complete = false; // every texel decoded: independent of VRAM from then on
    };
    uint32_t DecodeTexel(const GSDrawState &state, int u, int v, const uint16_t *clut);
    void SnapshotTexelSlot(TexelCacheSlot &slot, const GSDrawState &state);
    // Larger textures (screen copies) are not snapshotted; uploads into them flush the batch instead.
    static constexpr size_t kMaxSnapshotTexels = 256u * 256u;
    static constexpr size_t kTexelCacheSlots = 16u;
    // Returns the slot holding `state`'s texture, or nullptr when every slot is in use by the queued batch.
    TexelCacheSlot *PrepareTexelCache(const GSDrawState &state);
    // Runs row(y) for y in [y0, y1]; splits large primitives across worker threads when every pixel
    // write lands in its own bytes (16/32-bit frame and Z formats).
    void ForEachRow(const GSDrawState &state, int y0, int y1, int width, const std::function<void(int)> &row);
    // Consecutive triangles/sprites sharing a texture are queued and rasterized together by row bands
    // across the worker pool; flushed before any other VRAM, CLUT or transfer operation.
    static constexpr size_t kMaxDeferredPrimitives = 512u;
    std::vector<GSPrimitiveBatch> m_deferred;

    bool CanDefer(const GSPrimitiveBatch &batch) const;
    bool m_deferralEnabled = false;

public:
    // Batched (banded, reordered) rasterization: opt-in with PS2X_GS_BATCH=1 until it passes PS2X_GS_SHADOW_CHECK
    // everywhere (a rare texture-page mismatch remains); always off for the shadow reference GS.
    void setDeferralEnabled(bool enabled) { m_deferralEnabled = enabled; }

private:
    // Whether queued primitives read (texture) or write (frame/Z) any VRAM page in [firstPage, endPage).
    bool DeferredTouchesPages(uint32_t firstPage, uint32_t endPage, bool includeTextures,
                              bool includeTargets = true) const;
    // Flushes the queue when a host->local or local->local transfer conflicts with it.
    void FlushDeferredForTransferUnlocked(const GSTransferCommand &command);
    void FlushDeferredUnlocked();
    std::array<TexelCacheSlot, kTexelCacheSlots> m_texelSlots;
    uint64_t m_texelSlotClock = 0u;
    uint32_t m_batchSlotMask = 0u;           // slots referenced by queued primitives
    std::vector<int8_t> m_deferredSlots;     // slot of each queued primitive, -1 when untextured
    const GSDrawState *m_texelCacheOwner = nullptr; // primitive drawn outside a batch, using m_serialSlot
    TexelCacheSlot *m_serialSlot = nullptr;
    // Texel-cache slot of the queued primitive this thread is rasterizing (nullptr outside batches).
    static thread_local TexelCacheSlot *t_texelSlot;
    uint32_t m_clutGeneration = 0u;

    GSTransferCommand m_transfer{};
    GSTransferSnapshot m_transferState{};
    std::vector<uint8_t> m_localToHostBuffer;
    size_t m_localToHostReadPos = 0;
};
