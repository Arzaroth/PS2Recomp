#ifndef PS2_GS_TRACE_H
#define PS2_GS_TRACE_H

// GS command streams recorded in a game with PS2X_GS_RECORD and replayed by the ps2_gs_replay tool, to time the
// rasterizer and check that a change leaves VRAM bit for bit the same.
//   header: magic, VRAM size, VRAM, drawing state size, drawing state (GS::saveDrawingState)
//   then commands: kind (GifPacket / NativePackedGifPacket / NativeImageUpload), four registers, payload
#include <cstdint>
#include <cstdio>
#include <vector>

struct GsTraceCommand
{
    enum Kind : uint8_t
    {
        GifPacket = 0,
        NativePackedGifPacket = 1,
        NativeImageUpload = 2
    };
    uint8_t kind = GifPacket;
    uint64_t registers[4]{};
    std::vector<uint8_t> bytes;
};

struct GsTraceHeader
{
    static constexpr uint32_t kMagic = 0x31525347u; // "GSR1"
    std::vector<uint8_t> vram;
    std::vector<uint8_t> state;
};

inline void writeGsTraceHeader(FILE *out, const GsTraceHeader &header)
{
    const uint32_t magic = GsTraceHeader::kMagic;
    const uint32_t vramSize = static_cast<uint32_t>(header.vram.size());
    const uint32_t stateSize = static_cast<uint32_t>(header.state.size());
    std::fwrite(&magic, 4, 1, out);
    std::fwrite(&vramSize, 4, 1, out);
    std::fwrite(header.vram.data(), 1, header.vram.size(), out);
    std::fwrite(&stateSize, 4, 1, out);
    std::fwrite(header.state.data(), 1, header.state.size(), out);
}

inline void writeGsTraceCommand(FILE *out, uint8_t kind, const uint64_t registers[4], const uint8_t *data, uint32_t size)
{
    std::fwrite(&kind, 1, 1, out);
    std::fwrite(registers, 8, 4, out);
    std::fwrite(&size, 4, 1, out);
    if (size != 0u)
        std::fwrite(data, 1, size, out);
}

inline bool readGsTraceHeader(FILE *in, GsTraceHeader &header)
{
    uint32_t magic = 0, vramSize = 0, stateSize = 0;
    if (std::fread(&magic, 4, 1, in) != 1 || magic != GsTraceHeader::kMagic || std::fread(&vramSize, 4, 1, in) != 1)
        return false;
    header.vram.resize(vramSize);
    if (std::fread(header.vram.data(), 1, vramSize, in) != vramSize || std::fread(&stateSize, 4, 1, in) != 1)
        return false;
    header.state.resize(stateSize);
    return std::fread(header.state.data(), 1, stateSize, in) == stateSize;
}

inline bool readGsTraceCommand(FILE *in, GsTraceCommand &command)
{
    uint32_t size = 0;
    if (std::fread(&command.kind, 1, 1, in) != 1 || std::fread(command.registers, 8, 4, in) != 4 ||
        std::fread(&size, 4, 1, in) != 1)
        return false;
    command.bytes.resize(size);
    return size == 0u || std::fread(command.bytes.data(), 1, size, in) == size;
}

#endif
