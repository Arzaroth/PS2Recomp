#pragma once

#include "runtime/gs/gs_backend.h"
#include "runtime/gs/gs_texture_page_cache.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <map>
#include <string>
#include <tuple>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

class GSCpuBackend final : public GSRasterBackend
{
public:
    GSCpuBackend();
    ~GSCpuBackend() override;

    // Opt-in, process wide: backends draw on worker threads, asynchronously to Submit. Code
    // that reads VRAM must then go through the backend (which waits for queued drawing);
    // raw reads of the VRAM buffer may see primitives still being drawn.
    static void SetAsyncRasterization(bool enabled) noexcept;
    // Keeps this backend synchronous even when asynchronous rasterization is enabled globally (the
    // PS2X_GS_SHADOW_CHECK reference GS).
    void DisallowAsyncRasterization() noexcept { m_asyncAllowed = false; }

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
    bool m_asyncAllowed = true;
    void ResetUnlocked();
    void LoadClutUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut);
    uint32_t ReadVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const;
    uint32_t ReadTextureVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y);
    void WriteVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value);

    // Registers decoded once per primitive, and the slice of rows one thread rasterizes.
    struct PixelPipeline;
    struct TextureSampler;
    struct RasterContext;
    PixelPipeline MakePixelPipeline(const GSDrawState &state) const;
    TextureSampler MakeTextureSampler(const GSDrawState &state) const;
    const std::array<uint16_t, 512> *ClutSnapshotUnlocked(uint64_t queueIndex);

    void DrawPrimitive(const GSPrimitiveBatch &batch);
    void DrawSprite(const GSPrimitiveBatch &batch, const RasterContext &raster);
    void DrawTriangle(const GSPrimitiveBatch &batch, const RasterContext &raster);
    void DrawLine(const GSPrimitiveBatch &batch, const RasterContext &raster);
    void WritePixel(const PixelPipeline &pipeline, int x, int y, uint32_t z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog);
    uint32_t SampleTexture(const TextureSampler &sampler, GSMem::TexturePageCache &cache, float s, float t, float q, uint16_t u, uint16_t v);

    // Submit queues primitives; worker threads draw them in order, each on the VRAM pages it
    // owns, so every word sees the same write order as a serial rasterizer. Anything else
    // that touches VRAM drains the queue first.
    struct QueuedPrimitive;
    struct DecodedTexture;
    struct DecodedTextureCache;
    uint32_t FetchTexel(const TextureSampler &sampler, GSMem::TexturePageCache &cache, int u, int v);
    void DecodeTile(DecodedTexture &texture, const TextureSampler &sampler, GSMem::TexturePageCache &cache, uint32_t tile);
    DecodedTexture *FindDecodedTexture(DecodedTextureCache &cache, const TextureSampler &sampler);
    struct alignas(64) WorkerProgress
    {
        std::atomic<uint64_t> done{0};
    };
    void StartWorkersUnlocked();
    void DrainUnlocked() const;
    // VRAM written by the caller after a drain: the next primitive sampling it gets a barrier,
    // so workers drop decoded textures that predate the write.
    void NoteDirectWrite(std::pair<uint32_t, uint32_t> blocks);
    void NotePendingRange(std::vector<std::pair<uint32_t, uint32_t>> &ranges, std::pair<uint32_t, uint32_t> range);
    void NoteBarrierRange(std::vector<std::pair<uint32_t, uint32_t>> &ranges, std::pair<uint32_t, uint32_t> range);
    // Writes (or, with write false, only counts) the pixels of an upload chunk; with rows, only
    // those on the rows that raster slice owns. Returns the pixels consumed.
    uint32_t WriteUploadPixels(const GSTransferCommand &transfer, uint32_t firstPixel, uint32_t totalPixels,
                               const uint8_t *data, uint32_t sizeBytes, const RasterContext *rows, bool write);
    QueuedPrimitive &ReserveQueueSlotUnlocked();
    void PublishQueueSlotUnlocked();
    void DrawQueued(const QueuedPrimitive &item, uint32_t index);
    void DrawWith(const GSPrimitiveBatch &batch, const RasterContext &raster);
    void WorkerMain(uint32_t index);

    // PS2X_GS_DUMP=<dir>, PS2X_GS_DUMP_AFTER=<seconds>: after that delay, the next 4000 (PS2X_GS_DUMP_PRIMS)
    // primitives are drawn immediately and logged with their state (prims.txt), each distinct
    // texture decoded as sampled (tex_N_WxH.rgba), with VRAM before and after, and the render
    // target after each group of primitives until the PS2X_GS_DUMP_SNAP_PRESENTS-th present (default 1).
    void DumpUpdateUnlocked();
    void DumpLog(const char *format, ...) const;
    void DumpPrimitiveUnlocked(const GSPrimitiveBatch &batch);
    void DumpFinishUnlocked();
    uint32_t LookupCLUT(const GSDrawState &state, const std::array<uint16_t, 512> &clut, uint8_t index, uint8_t cpsm, uint8_t csm, uint8_t csa, uint8_t sourcePsm);

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
    // Copies of m_clut taken when it changed, one per run of queued primitives that sample it.
    std::vector<std::array<uint16_t, 512>> m_clutSnapshots;
    std::vector<uint64_t> m_clutSnapshotLastUse; // queue index + 1 of the slot's last user
    uint32_t m_clutSnapshotSlot = 0;
    bool m_clutSnapshotStale = true;
    std::vector<GSMem::TexturePageCache> m_texturePageCaches; // [0] caller (CLUT loads), [1 + i] worker i
    std::vector<DecodedTextureCache> m_decodedCaches;           // same indexing, used by workers only
    uint64_t m_clutSerial = 1;                                  // bumped whenever m_clut changes

    std::vector<QueuedPrimitive> m_queue; // ring of primitives
    std::atomic<uint64_t> m_queueTail{0}; // primitives queued so far
    std::unique_ptr<WorkerProgress[]> m_workerProgress;
    uint32_t m_workerCount = 0;
    // VRAM block ranges the queued primitives draw to (start, count): a CLUT load from one of
    // them waits for the queue to drain.
    mutable std::vector<std::pair<uint32_t, uint32_t>> m_pendingWrites;
    // Writes queued since the last in-queue barrier: a primitive sampling one of them gets a
    // barrier (the workers wait for each other there) instead of draining the queue. Pixel
    // writes (color, uploads) are made by the owner of their page, depth writes by the owner of
    // the color page, so the two kinds overlapping also need a barrier.
    std::vector<std::pair<uint32_t, uint32_t>> m_pixelWritesSinceBarrier;
    std::vector<std::pair<uint32_t, uint32_t>> m_zWritesSinceBarrier;
    // Texture ranges read since the last barrier: a queued write over them gets a barrier.
    mutable std::vector<std::pair<uint32_t, uint32_t>> m_readsSinceBarrier;
    bool m_barrierRequested = false;
    // Queued items wrote VRAM since m_texturePageCaches[0] (CLUT loads) was last emptied.
    bool m_callerCacheStale = false;
    // Z buffers written since the last barrier, with the color-page layout that decided
    // which worker wrote each depth word.
    mutable std::vector<uint64_t> m_zOwnerKeys;
    std::vector<std::thread> m_workers;
    std::mutex m_workMutex;
    std::condition_variable m_workCv;
    std::atomic<uint32_t> m_sleepingWorkers{0};
    std::atomic<bool> m_stopWorkers{false};
    uint32_t m_dumpState = 0; // 0 unchecked, 1 armed, 2 done or disabled
    uint32_t m_dumpRemaining = 0;
    uint32_t m_dumpTotal = 0;
    std::string m_dumpDir;
    std::chrono::steady_clock::time_point m_dumpStart{};
    FILE *m_dumpLog = nullptr;
    mutable uint32_t m_dumpPresents = 0;
    uint32_t m_dumpSnapshots = 0;
    uint64_t m_dumpGroup = 0;
    bool m_dumpGroupValid = false;
    GSFrameReg m_dumpGroupFrame{};
    uint32_t m_dumpGroupScissorX1 = 0;
    uint32_t m_dumpGroupScissorY1 = 0;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, int, int, uint32_t, uint32_t, uint32_t, uint64_t, uint32_t, uint32_t, uint32_t>, int> m_dumpTextures;
    static std::atomic<bool> s_asyncRasterization;

    GSTransferCommand m_transfer{};
    GSTransferSnapshot m_transferState{};
    std::vector<uint8_t> m_localToHostBuffer;
    size_t m_localToHostReadPos = 0;
};
