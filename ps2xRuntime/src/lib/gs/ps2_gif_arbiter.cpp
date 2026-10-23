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
        std::atomic<uint64_t> g_formats[4][4]{};
    }

    void notePacket(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes)
    {
        const size_t index = static_cast<size_t>(pathId) & 3u;
        if (data && sizeBytes >= 16u)
        {
            uint64_t tagLo = 0;
            std::memcpy(&tagLo, data, sizeof(tagLo));
            g_formats[index][(tagLo >> 58u) & 3u].fetch_add(1u, std::memory_order_relaxed);
        }
        g_packets[index].fetch_add(1u, std::memory_order_relaxed);
        g_bytes[index].fetch_add(sizeBytes, std::memory_order_relaxed);
    }

    uint64_t packets(GifPathId pathId)
    {
        return g_packets[static_cast<size_t>(pathId) & 3u].load(std::memory_order_relaxed);
    }

    uint64_t tagFormats(GifPathId pathId, uint32_t format)
    {
        return g_formats[static_cast<size_t>(pathId) & 3u][format & 3u].load(std::memory_order_relaxed);
    }

    uint64_t bytes(GifPathId pathId)
    {
        return g_bytes[static_cast<size_t>(pathId) & 3u].load(std::memory_order_relaxed);
    }
}

void GifArbiter::submit(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool path2DirectHl, bool continueTags)
{
    if (!data || sizeBytes < 16 || !m_processFn)
        return;

    ps2_gif_path_stats::notePacket(pathId, data, sizeBytes);
    GifArbiterPacket pkt;
    if (continueTags)
        pkt.data = completeTags(pathId, data, sizeBytes);
    else
        pkt.data.assign(data, data + sizeBytes);
    if (pkt.data.size() < 16u)
        return;
    pkt.pathId = pathId;
    pkt.path2DirectHl = (pathId == GifPathId::Path2) && path2DirectHl;
    pkt.path3Image = (pathId == GifPathId::Path3) && isImagePacket(pkt.data.data(), static_cast<uint32_t>(pkt.data.size()));
    m_queue.push_back(std::move(pkt));
}

std::vector<uint8_t> GifArbiter::completeTags(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes)
{
    PathContinuation &pending = m_continuations[static_cast<size_t>(pathId) & 3u];
    std::vector<uint8_t> packet;
    packet.reserve(sizeBytes + pending.bytes.size() + 16u);
    if (pending.loopsLeft != 0u)
    {
        constexpr uint64_t kNloopMask = 0x7FFFull;
        constexpr uint64_t kPre = 1ull << 46;
        const uint64_t tagLo = (pending.tagLo & ~(kNloopMask | kPre)) | pending.loopsLeft;
        packet.resize(16u);
        std::memcpy(packet.data(), &tagLo, 8u);
        std::memcpy(packet.data() + 8u, &pending.tagHi, 8u);
        packet.insert(packet.end(), pending.bytes.begin(), pending.bytes.end());
    }
    packet.insert(packet.end(), data, data + sizeBytes);
    pending.loopsLeft = 0u;
    pending.bytes.clear();

    size_t offset = 0;
    while (offset + 16u <= packet.size())
    {
        uint64_t tagLo = 0;
        uint64_t tagHi = 0;
        std::memcpy(&tagLo, packet.data() + offset, 8u);
        std::memcpy(&tagHi, packet.data() + offset + 8u, 8u);
        const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
        const uint32_t flg = static_cast<uint32_t>((tagLo >> 58) & 0x3u);
        uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
        if (nreg == 0u)
            nreg = 16u;
        const size_t loopBytes = flg == 0u ? nreg * 16u : flg == 1u ? nreg * 8u : 16u;
        const size_t dataBytes = flg == 1u ? ((static_cast<size_t>(nloop) * nreg + 1u) & ~size_t{1}) * 8u : nloop * loopBytes;
        if (offset + 16u + dataBytes <= packet.size())
        {
            offset += 16u + dataBytes;
            continue;
        }
        // The tag continues in the path's next transfer. A REGLIST cut must fall on a whole quadword.
        uint32_t loops = static_cast<uint32_t>((packet.size() - offset - 16u) / loopBytes);
        if (flg == 1u && loops != 0u && ((loops * nreg) & 1u) != 0u)
            --loops;
        const size_t cut = offset + 16u + loops * loopBytes;
        pending.tagLo = tagLo;
        pending.tagHi = tagHi;
        pending.loopsLeft = nloop - loops;
        pending.bytes.assign(packet.begin() + static_cast<std::ptrdiff_t>(cut), packet.end());
        if (loops == 0u)
        {
            packet.resize(offset);
            break;
        }
        const uint64_t partLo = (tagLo & ~(0x7FFFull | (1ull << 15))) | loops;
        std::memcpy(packet.data() + offset, &partLo, 8u);
        packet.resize(cut);
        break;
    }
    return packet;
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
