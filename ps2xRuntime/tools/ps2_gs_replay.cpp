// Replays a GS command stream recorded with PS2X_GS_RECORD and prints the time per pass and a hash of the final
// VRAM, to time the rasterizer and check that a change draws the same pixels.
//   ps2_gs_replay <trace> [repeat] [out.rgba]
// PS2X_GS_ASYNC_RASTER=1 (and PS2X_GS_RASTER_THREADS=<n>) replays on the rasterization workers.
// PS2X_GS_REPLAY_COMMANDS=<n> replays only the first n commands.
// PS2X_GS_REPLAY_CHECK=<n> replays on the workers and on the caller side by side, compares their VRAM every n commands
// and reports the first comparison that differs.
#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/gs_trace.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    void replayCommand(GS &gs, const GsTraceCommand &command)
    {
        const uint8_t *data = command.bytes.data();
        const uint32_t size = static_cast<uint32_t>(command.bytes.size());
        switch (command.kind)
        {
        case GsTraceCommand::GifPacket:
            gs.processGIFPacket(data, size);
            break;
        case GsTraceCommand::NativePackedGifPacket:
            gs.processNativePackedGIFPacket(data, size);
            break;
        case GsTraceCommand::NativeImageUpload:
            gs.uploadImageNative(command.registers[0], command.registers[1], command.registers[2], command.registers[3],
                                 data, size);
            break;
        }
    }

    int checkWorkersAgainstCaller(const GsTraceHeader &header, const std::vector<GsTraceCommand> &commands, size_t interval)
    {
        GSCpuBackend::SetAsyncRasterization(true);
        std::vector<uint8_t> asyncVram(header.vram), serialVram(header.vram);
        GS async, serial;
        async.init(asyncVram.data(), static_cast<uint32_t>(asyncVram.size()), nullptr);
        serial.init(serialVram.data(), static_cast<uint32_t>(serialVram.size()), nullptr);
        if (auto *cpu = dynamic_cast<GSCpuBackend *>(serial.rasterBackend()))
            cpu->DisallowAsyncRasterization();
        if (!async.loadDrawingState(header.state) || !serial.loadDrawingState(header.state))
        {
            std::fprintf(stderr, "bad drawing state\n");
            return 1;
        }
        std::vector<uint8_t> asyncSnapshot, serialSnapshot;
        for (size_t index = 0; index < commands.size(); ++index)
        {
            replayCommand(async, commands[index]);
            replayCommand(serial, commands[index]);
            if ((index + 1u) % interval != 0u && index + 1u != commands.size())
                continue;
            async.rasterBackend()->SnapshotVram(asyncSnapshot);
            serial.rasterBackend()->SnapshotVram(serialSnapshot);
            if (asyncSnapshot == serialSnapshot)
                continue;
            size_t words = 0, first = 0;
            for (size_t offset = 0; offset + 4 <= serialSnapshot.size(); offset += 4)
                if (std::memcmp(asyncSnapshot.data() + offset, serialSnapshot.data() + offset, 4) != 0 && words++ == 0)
                    first = offset;
            std::printf("command %zu: %zu words differ, first at byte 0x%zx (page %zu)\n", index, words, first,
                        first / 8192u);
            return 1;
        }
        std::printf("workers match the caller over %zu commands\n", commands.size());
        return 0;
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s <trace> [repeat] [vram-out]\n", argv[0]);
        return 2;
    }
    FILE *in = std::fopen(argv[1], "rb");
    GsTraceHeader header;
    if (!in || !readGsTraceHeader(in, header))
    {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    std::vector<GsTraceCommand> commands;
    uint64_t bytes = 0;
    for (GsTraceCommand command; readGsTraceCommand(in, command);)
    {
        bytes += command.bytes.size();
        commands.push_back(std::move(command));
    }
    std::fclose(in);
    const int repeat = argc > 2 ? std::max(1, std::atoi(argv[2])) : 1;
    if (const char *limit = std::getenv("PS2X_GS_REPLAY_COMMANDS"))
        commands.resize(std::min<size_t>(commands.size(), std::strtoull(limit, nullptr, 0)));
    if (std::getenv("PS2X_GS_REPLAY_CHECK"))
        return checkWorkersAgainstCaller(header, commands, std::max<size_t>(1u, std::strtoull(std::getenv("PS2X_GS_REPLAY_CHECK"), nullptr, 0)));
    if (std::getenv("PS2X_GS_ASYNC_RASTER"))
        GSCpuBackend::SetAsyncRasterization(true);

    std::vector<uint8_t> vram(header.vram.size());
    std::vector<uint8_t> final;
    double best = 0.0, total = 0.0;
    for (int pass = 0; pass < repeat; ++pass)
    {
        std::memcpy(vram.data(), header.vram.data(), vram.size());
        GS gs;
        gs.init(vram.data(), static_cast<uint32_t>(vram.size()), nullptr);
        if (!gs.loadDrawingState(header.state))
        {
            std::fprintf(stderr, "bad drawing state\n");
            return 1;
        }
        const auto start = std::chrono::steady_clock::now();
        for (const GsTraceCommand &command : commands)
            replayCommand(gs, command);
        gs.rasterBackend()->SnapshotVram(final); // waits for the workers
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        total += seconds;
        best = pass == 0 ? seconds : std::min(best, seconds);
    }

    uint64_t hash = 0xCBF29CE484222325ull;
    for (const uint8_t byte : final)
        hash = (hash ^ byte) * 0x100000001B3ull;
    std::printf("commands=%zu bytes=%llu vram-hash=%016llx\n", commands.size(), static_cast<unsigned long long>(bytes),
                static_cast<unsigned long long>(hash));
    std::printf("time per pass: best %.1f ms, mean %.1f ms\n", best * 1000.0, total * 1000.0 / repeat);
    if (argc > 3)
        if (FILE *out = std::fopen(argv[3], "wb"))
        {
            std::fwrite(final.data(), 1, final.size(), out);
            std::fclose(out);
        }
    return 0;
}
