#ifndef PS2_VU1_TRACE_H
#define PS2_VU1_TRACE_H

#include "runtime/ps2_vu1.h"

#include <cstdint>
#include <cstdio>
#include <vector>

// Recorded VU1 microprogram runs (PS2X_VU1_RECORD) replayed by the ps2_vu1_replay tool to check and
// benchmark the interpreter against a previous build's results.
struct Vu1TraceRecord
{
    static constexpr uint32_t kMagic = 0x52315556u; // "VU1R"
    static constexpr uint32_t kMemorySize = 16u * 1024u;

    uint32_t resume = 0; // 0: MSCAL (execute), 1: MSCNT (resume)
    uint32_t startPC = 0;
    uint32_t top = 0;
    uint32_t itop = 0;
    uint32_t maxCycles = 0;
    std::vector<uint8_t> code; // present when the microcode changed since the previous record
    VU1State stateBefore{};
    std::vector<uint8_t> dataBefore;
    VU1State stateAfter{};
    std::vector<uint8_t> dataAfter;
    std::vector<std::vector<uint8_t>> packets;
};

inline void writeVu1TraceRecord(FILE *out, const Vu1TraceRecord &record)
{
    auto put = [out](const void *data, size_t size) { std::fwrite(data, 1, size, out); };
    auto putU32 = [&](uint32_t value) { put(&value, sizeof(value)); };
    putU32(Vu1TraceRecord::kMagic);
    putU32(record.resume);
    putU32(record.startPC);
    putU32(record.top);
    putU32(record.itop);
    putU32(record.maxCycles);
    putU32(record.code.empty() ? 0u : 1u);
    if (!record.code.empty())
        put(record.code.data(), Vu1TraceRecord::kMemorySize);
    put(&record.stateBefore, sizeof(VU1State));
    put(record.dataBefore.data(), Vu1TraceRecord::kMemorySize);
    put(&record.stateAfter, sizeof(VU1State));
    put(record.dataAfter.data(), Vu1TraceRecord::kMemorySize);
    putU32(static_cast<uint32_t>(record.packets.size()));
    for (const std::vector<uint8_t> &packet : record.packets)
    {
        putU32(static_cast<uint32_t>(packet.size()));
        put(packet.data(), packet.size());
    }
}

inline bool readVu1TraceRecord(FILE *in, Vu1TraceRecord &record)
{
    auto get = [in](void *data, size_t size) { return std::fread(data, 1, size, in) == size; };
    uint32_t magic = 0u, hasCode = 0u, packetCount = 0u;
    if (!get(&magic, 4) || magic != Vu1TraceRecord::kMagic)
        return false;
    if (!get(&record.resume, 4) || !get(&record.startPC, 4) || !get(&record.top, 4) || !get(&record.itop, 4) ||
        !get(&record.maxCycles, 4) || !get(&hasCode, 4))
        return false;
    record.code.clear();
    if (hasCode)
    {
        record.code.resize(Vu1TraceRecord::kMemorySize);
        if (!get(record.code.data(), Vu1TraceRecord::kMemorySize))
            return false;
    }
    record.dataBefore.resize(Vu1TraceRecord::kMemorySize);
    record.dataAfter.resize(Vu1TraceRecord::kMemorySize);
    if (!get(&record.stateBefore, sizeof(VU1State)) || !get(record.dataBefore.data(), Vu1TraceRecord::kMemorySize) ||
        !get(&record.stateAfter, sizeof(VU1State)) || !get(record.dataAfter.data(), Vu1TraceRecord::kMemorySize) ||
        !get(&packetCount, 4))
        return false;
    record.packets.assign(packetCount, {});
    for (std::vector<uint8_t> &packet : record.packets)
    {
        uint32_t size = 0u;
        if (!get(&size, 4))
            return false;
        packet.resize(size);
        if (!get(packet.data(), size))
            return false;
    }
    return true;
}

#endif // PS2_VU1_TRACE_H
