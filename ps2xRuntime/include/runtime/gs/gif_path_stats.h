#ifndef PS2_GIF_PATH_STATS_H
#define PS2_GIF_PATH_STATS_H

#include "runtime/gs/ps2_gif_arbiter.h"

#include <cstdint>

namespace ps2_gif_path_stats
{
    void notePacket(GifPathId pathId, uint32_t sizeBytes);
    uint64_t packets(GifPathId pathId);
    uint64_t bytes(GifPathId pathId);
}

#endif
