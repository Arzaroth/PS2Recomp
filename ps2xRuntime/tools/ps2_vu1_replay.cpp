// Replays VU1 microprogram runs recorded with PS2X_VU1_RECORD and compares the interpreter's results
// (registers, VU1 data memory, PATH1 packets, cycle count) with the recording.
//   ps2_vu1_replay <trace> [repeat]   repeat > 1 also reports the interpreter's speed.
//   ps2_vu1_replay <trace> stats      VU cycles per microprogram (code image and start pc).
//   ps2_vu1_replay <trace> dumpcode <dir>   writes each distinct code image as <dir>/<hash>.bin.
#include "runtime/gs/gs_frontend.h"
#include "runtime/ps2_vu1_trace.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

// ps2_runtime expects the generated runner to provide its function table; this tool runs no guest code.
struct R5900Context;
class PS2Runtime;
extern const uint32_t g_ps2RecompiledFunctionTableBase = 0u;
extern const uint32_t g_ps2RecompiledFunctionTableEnd = 0u;
extern const uint32_t g_ps2RecompiledFunctionTableSlotCount = 0u;
void (*g_ps2RecompiledFunctionTable[1])(uint8_t *, R5900Context *, PS2Runtime *) = {nullptr};

namespace
{
    std::string describeStateDifference(const VU1State &expected, const VU1State &actual)
    {
        std::string diff;
        char line[256];
        for (int reg = 0; reg < 32; ++reg)
            if (std::memcmp(expected.vf[reg], actual.vf[reg], 16) != 0)
            {
                uint32_t e[4], a[4];
                std::memcpy(e, expected.vf[reg], 16);
                std::memcpy(a, actual.vf[reg], 16);
                std::snprintf(line, sizeof(line), " vf%d exp=%08x,%08x,%08x,%08x got=%08x,%08x,%08x,%08x", reg, e[0], e[1],
                              e[2], e[3], a[0], a[1], a[2], a[3]);
                diff += line;
            }
        for (int reg = 0; reg < 16; ++reg)
            if (expected.vi[reg] != actual.vi[reg])
            {
                std::snprintf(line, sizeof(line), " vi%d exp=%d got=%d", reg, expected.vi[reg], actual.vi[reg]);
                diff += line;
            }
        auto scalar = [&](const char *name, const void *e, const void *a, size_t size)
        {
            if (std::memcmp(e, a, size) != 0)
            {
                uint32_t ev = 0, av = 0;
                std::memcpy(&ev, e, size < 4 ? size : 4);
                std::memcpy(&av, a, size < 4 ? size : 4);
                std::snprintf(line, sizeof(line), " %s exp=%08x got=%08x", name, ev, av);
                diff += line;
            }
        };
        scalar("acc", expected.acc, actual.acc, 16);
        scalar("q", &expected.q, &actual.q, 4);
        scalar("p", &expected.p, &actual.p, 4);
        scalar("i", &expected.i, &actual.i, 4);
        scalar("r", &expected.r, &actual.r, 4);
        scalar("pc", &expected.pc, &actual.pc, 4);
        scalar("mac", &expected.mac, &actual.mac, 4);
        scalar("clip", &expected.clip, &actual.clip, 4);
        scalar("status", &expected.status, &actual.status, 4);
        scalar("stoppedByD", &expected.stoppedByD, &actual.stoppedByD, 1);
        scalar("stoppedByT", &expected.stoppedByT, &actual.stoppedByT, 1);
        return diff;
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s <trace> [repeat]\n", argv[0]);
        return 2;
    }
    FILE *in = std::fopen(argv[1], "rb");
    if (!in)
    {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    std::vector<Vu1TraceRecord> records;
    for (Vu1TraceRecord record; readVu1TraceRecord(in, record);)
        records.push_back(record);
    std::fclose(in);
    if (argc > 3 && std::string_view(argv[2]) == "dumpcode")
    {
        std::map<size_t, const std::vector<uint8_t> *> images;
        for (const Vu1TraceRecord &record : records)
            if (!record.code.empty())
                images.emplace(std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char *>(record.code.data()), record.code.size())),
                               &record.code);
        for (const auto &[hash, code] : images)
        {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/%016zx.bin", argv[3], hash);
            if (FILE *out = std::fopen(path, "wb"))
            {
                std::fwrite(code->data(), 1, code->size(), out);
                std::fclose(out);
            }
            std::printf("%s\n", path);
        }
        return 0;
    }
    if (argc > 2 && std::string_view(argv[2]) == "stats")
    {
        struct Program
        {
            uint64_t runs = 0;
            uint64_t cycles = 0;
        };
        std::map<std::pair<size_t, uint32_t>, Program> programs;
        size_t codeHash = 0;
        uint64_t total = 0;
        for (const Vu1TraceRecord &record : records)
        {
            if (!record.code.empty())
                codeHash = std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char *>(record.code.data()), record.code.size()));
            Program &program = programs[{codeHash, record.startPC}];
            const uint64_t cycles = record.stateAfter.cycles - record.stateBefore.cycles;
            ++program.runs;
            program.cycles += cycles;
            total += cycles;
        }
        std::vector<std::pair<std::pair<size_t, uint32_t>, Program>> sorted(programs.begin(), programs.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) { return a.second.cycles > b.second.cycles; });
        std::printf("programs=%zu runs=%zu vu-cycles=%llu\n", sorted.size(), records.size(), static_cast<unsigned long long>(total));
        for (size_t i = 0; i < sorted.size() && i < 40u; ++i)
            std::printf("code=%016zx pc=0x%04x runs=%llu cycles=%llu (%.1f%%) avg=%llu\n", sorted[i].first.first, sorted[i].first.second,
                        static_cast<unsigned long long>(sorted[i].second.runs), static_cast<unsigned long long>(sorted[i].second.cycles),
                        100.0 * static_cast<double>(sorted[i].second.cycles) / static_cast<double>(std::max<uint64_t>(total, 1u)),
                        static_cast<unsigned long long>(sorted[i].second.cycles / std::max<uint64_t>(sorted[i].second.runs, 1u)));
        return 0;
    }
    const int repeat = argc > 2 ? std::atoi(argv[2]) : 1;

    GS gs; // never initialized: packets are only observed
    VU1Interpreter vu1;
    std::vector<uint8_t> code(Vu1TraceRecord::kMemorySize), data(Vu1TraceRecord::kMemorySize);
    std::vector<std::vector<uint8_t>> packets;
    vu1.setPacketObserver([&packets](const uint8_t *packet, uint32_t size)
                          { packets.emplace_back(packet, packet + size); });

    size_t mismatches = 0, resumeMismatches = 0, timingMismatches = 0;
    uint64_t totalCycles = 0;
    double seconds = 0.0;
    for (int pass = 0; pass < repeat; ++pass)
    {
        const auto start = std::chrono::steady_clock::now();
        for (size_t index = 0; index < records.size(); ++index)
        {
            const Vu1TraceRecord &record = records[index];
            if (!record.code.empty())
            {
                code = record.code;
                vu1.invalidateDecodedCode();
            }
            const uint64_t cyclesBefore = vu1.state().cycles; // the interpreter's own clock after the previous run
            vu1.state() = record.stateBefore;
            data = record.dataBefore;
            packets.clear();
            if (record.resume)
                vu1.resume(code.data(), Vu1TraceRecord::kMemorySize, data.data(), Vu1TraceRecord::kMemorySize, gs, nullptr,
                           record.top, record.itop, record.maxCycles);
            else
                vu1.execute(code.data(), Vu1TraceRecord::kMemorySize, data.data(), Vu1TraceRecord::kMemorySize, gs,
                            nullptr, record.startPC, record.top, record.itop, record.maxCycles);
            if (pass != 0)
                continue;

            const VU1State &actual = vu1.state();
            const uint64_t expectedCycles = record.stateAfter.cycles - record.stateBefore.cycles;
            const uint64_t actualCycles = actual.cycles - cyclesBefore;
            totalCycles += expectedCycles;
            std::string diff = describeStateDifference(record.stateAfter, actual);
            const bool dataDiffers = data != record.dataAfter;
            const bool packetsDiffer = packets != record.packets;
            if (diff.empty() && !dataDiffers && !packetsDiffer)
            {
                if (expectedCycles != actualCycles && timingMismatches++ < 10)
                    std::printf("#%zu pc=0x%x cycles exp=%llu got=%llu\n", index, record.startPC,
                                static_cast<unsigned long long>(expectedCycles),
                                static_cast<unsigned long long>(actualCycles));
                continue;
            }
            size_t &counter = record.resume ? resumeMismatches : mismatches;
            if (counter++ < 10)
            {
                std::printf("#%zu %s pc=0x%x:%s%s%s\n", index, record.resume ? "resume" : "mscal", record.startPC,
                            diff.c_str(), dataDiffers ? " data" : "", packetsDiffer ? " packets" : "");
                if (packetsDiffer)
                    std::printf("   packets exp=%zu got=%zu\n", record.packets.size(), packets.size());
                if (dataDiffers)
                    for (size_t offset = 0; offset < data.size(); offset += 4)
                        if (std::memcmp(&data[offset], &record.dataAfter[offset], 4) != 0)
                        {
                            uint32_t e, a;
                            std::memcpy(&e, &record.dataAfter[offset], 4);
                            std::memcpy(&a, &data[offset], 4);
                            std::printf("   first data diff at 0x%zx exp=%08x got=%08x\n", offset, e, a);
                            break;
                        }
            }
        }
        if (pass != 0 || repeat == 1)
            seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    std::printf("records=%zu mismatches=%zu resume-mismatches=%zu timing-only=%zu vu-cycles=%llu\n", records.size(),
                mismatches, resumeMismatches, timingMismatches, static_cast<unsigned long long>(totalCycles));
    const int timedPasses = repeat > 1 ? repeat - 1 : 1;
    std::printf("time per pass=%.2f ms  (%.1f ns per VU cycle)\n", seconds * 1000.0 / timedPasses,
                totalCycles ? seconds * 1e9 / timedPasses / static_cast<double>(totalCycles) : 0.0);
    return mismatches == 0 ? 0 : 1;
}
