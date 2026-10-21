#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/gs/gif_path_stats.h"
#include <algorithm>
#include <atomic>
#include <cstring>

GifArbiter::GifArbiter(ProcessPacketFn processFn)
    : m_processFn(std::move(processFn))
{
}

bool GifArbiter::isImagePacket(const uint8_t *data, uint32_t sizeBytes)
{
    if (!data || sizeBytes < 16u)
        return false;

    uint64_t tagLo = 0;
    std::memcpy(&tagLo, data, sizeof(tagLo));
    const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3u);
    return flg >= 2u; // IMAGE, or FLG 3 which behaves as IMAGE
}

extern std::atomic<uint32_t> g_gsDumpGifPath;

namespace ps2_gif_path_stats
{
    namespace
    {
        std::atomic<uint64_t> g_packets[4]{};
        std::atomic<uint64_t> g_bytes[4]{};
    }

    void notePacket(GifPathId pathId, uint32_t sizeBytes)
    {
        const size_t index = static_cast<size_t>(pathId) & 3u;
        g_packets[index].fetch_add(1u, std::memory_order_relaxed);
        g_bytes[index].fetch_add(sizeBytes, std::memory_order_relaxed);
    }

    uint64_t packets(GifPathId pathId)
    {
        return g_packets[static_cast<size_t>(pathId) & 3u].load(std::memory_order_relaxed);
    }

    uint64_t bytes(GifPathId pathId)
    {
        return g_bytes[static_cast<size_t>(pathId) & 3u].load(std::memory_order_relaxed);
    }
}

void GifArbiter::submit(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool path2DirectHl)
{
    if (!data || sizeBytes < 16 || !m_processFn)
        return;

    ps2_gif_path_stats::notePacket(pathId, sizeBytes);
    GifArbiterPacket pkt;
    pkt.pathId = pathId;
    pkt.path2DirectHl = (pathId == GifPathId::Path2) && path2DirectHl;
    pkt.path3Image = (pathId == GifPathId::Path3) && isImagePacket(data, sizeBytes);
    pkt.data.resize(sizeBytes);
    std::memcpy(pkt.data.data(), data, sizeBytes);
    m_queue.push_back(std::move(pkt));
}

void GifArbiter::drain()
{
    if (!m_processFn)
        return;

    std::stable_sort(m_queue.begin(), m_queue.end(),
                     [](const GifArbiterPacket &a, const GifArbiterPacket &b)
                     {
                         // DIRECTHL cannot preempt PATH3 IMAGE transfers.
                         if (a.path2DirectHl != b.path2DirectHl || a.path3Image != b.path3Image)
                         {
                             if (a.path3Image && b.path2DirectHl)
                                 return true;
                             if (a.path2DirectHl && b.path3Image)
                                 return false;
                         }
                         return pathPriority(a.pathId) < pathPriority(b.pathId);
                     });

    for (size_t i = 0; i < m_queue.size(); ++i)
    {
        auto &pkt = m_queue[i];
        if (!pkt.data.empty())
        {
            g_gsDumpGifPath.store(static_cast<uint32_t>(pkt.pathId), std::memory_order_relaxed);
            m_processFn(pkt.data.data(), static_cast<uint32_t>(pkt.data.size()));
            g_gsDumpGifPath.store(0u, std::memory_order_relaxed);
        }
    }
    m_queue.clear();
}

uint8_t GifArbiter::pathPriority(GifPathId id)
{
    return static_cast<uint8_t>(id);
}
