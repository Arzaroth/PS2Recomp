#ifndef PS2_GIF_ARBITER_H
#define PS2_GIF_ARBITER_H

#include <cstdint>
#include <functional>
#include <vector>

enum class GifPathId : uint8_t
{
    Path1 = 1,
    Path2 = 2,
    Path3 = 3,
};

struct GifArbiterPacket
{
    GifPathId pathId;
    bool path2DirectHl = false;
    bool path3Image = false;
    std::vector<uint8_t> data;
};

class GifArbiter
{
public:
    using ProcessPacketFn = std::function<void(const uint8_t *, uint32_t)>;

    GifArbiter() = default;
    explicit GifArbiter(ProcessPacketFn processFn);

    void setProcessPacketFn(ProcessPacketFn fn) { m_processFn = std::move(fn); }

    // With continueTags, a GIF tag the data ends in the middle of continues in the path's next such submission,
    // as across the DMA transfers of one path.
    void submit(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool path2DirectHl = false, bool continueTags = false);

    void drain();
    bool empty() const { return m_queue.empty(); }

private:
    // A GIF tag a transfer ended in the middle of: the path's next transfer carries the rest of its loops.
    struct PathContinuation
    {
        uint64_t tagLo = 0;
        uint64_t tagHi = 0;
        uint32_t loopsLeft = 0;
        std::vector<uint8_t> bytes; // data after the last complete loop sent
    };

    ProcessPacketFn m_processFn;
    std::vector<GifArbiterPacket> m_queue;
    PathContinuation m_continuations[4];

    // The packet with the path's pending continuation in front, cut after the last whole loop.
    std::vector<uint8_t> completeTags(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes);

    static bool isImagePacket(const uint8_t *data, uint32_t sizeBytes);
    static uint8_t pathPriority(GifPathId id);
};

#endif
