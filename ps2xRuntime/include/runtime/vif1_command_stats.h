#ifndef PS2_VIF1_COMMAND_STATS_H
#define PS2_VIF1_COMMAND_STATS_H

#include <cstdint>

namespace ps2_vif1_command_stats
{
    void noteCommand(uint8_t opcode);
    uint64_t commands(uint8_t opcode);
}

#endif
