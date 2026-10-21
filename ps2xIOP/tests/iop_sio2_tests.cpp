// SIO2 controller and DualShock 2, driven through the hardware registers as sio2man does.
#include "emulator/core/iop_memory.h"
#include "emulator/core/iop_sio2.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using namespace ps2x::iop::detail;

namespace
{
    constexpr uint32_t kSend3 = 0x1F808200u;
    constexpr uint32_t kDataIn = 0x1F808260u;
    constexpr uint32_t kDataOut = 0x1F808264u;
    constexpr uint32_t kCtrl = 0x1F808268u;
    constexpr uint32_t kRecv1 = 0x1F80826Cu;
    constexpr uint32_t kIstat = 0x1F808280u;

    bool expect(bool value, const char *message)
    {
        if (!value)
            std::cerr << "FAIL: " << message << '\n';
        return value;
    }

    // One transfer on a port through the FIFOs: returns the reply bytes.
    std::vector<uint8_t> transfer(IopMemory &memory, uint32_t port, const std::vector<uint8_t> &command)
    {
        const uint32_t size = static_cast<uint32_t>(command.size());
        memory.write32(kSend3, port | (size << 8) | (size << 18));
        memory.write32(kSend3 + 4u, 0u);
        for (const uint8_t byte : command)
            memory.write8(kDataIn, byte);
        memory.write32(kCtrl, 0x3BDu);
        std::vector<uint8_t> reply;
        for (uint32_t i = 0; i < size; ++i)
            reply.push_back(memory.read8(kDataOut));
        return reply;
    }
}

int main()
{
    IopMemory memory;
    IopSio2 sio2(memory);
    memory.setSio2(&sio2);
    uint16_t buttons = 0xFFFFu;
    sio2.setPadStateReader([&](uint8_t *state, size_t size)
                           {
        if (size < 8u)
            return false;
        state[2] = static_cast<uint8_t>(buttons);
        state[3] = static_cast<uint8_t>(buttons >> 8);
        state[4] = 0x11u; state[5] = 0x22u; state[6] = 0x33u; state[7] = 0x44u;
        return true; });

    // Power-on: digital mode, buttons from the host (cross held).
    buttons = static_cast<uint16_t>(~0x4000u);
    std::vector<uint8_t> reply = transfer(memory, 0u, {0x01, 0x42, 0x00, 0x00, 0x00});
    if (!expect(reply == std::vector<uint8_t>{0xFF, 0x41, 0x5A, 0xFF, 0xBF}, "digital poll reply")) return 1;
    if (!expect(memory.read32(kRecv1) == 0x1100u, "RECV1 must report a connected pad")) return 1;
    if (!expect((memory.read32(kIstat) & 1u) != 0u, "a finished transfer sets ISTAT")) return 1;
    const auto interrupt = memory.takeDmaStart();
    if (!expect(interrupt && interrupt->irq == IopSio2::Irq, "a finished transfer raises interrupt 17")) return 1;
    memory.write32(kIstat, 1u);
    if (!expect(memory.read32(kIstat) == 0u, "ISTAT acknowledges on write")) return 1;

    // Enter configuration, switch to analog and lock, query the model, exit.
    reply = transfer(memory, 0u, {0x01, 0x43, 0x00, 0x01, 0x00});
    if (!expect(reply[1] == 0x41u, "entering configuration still answers as a poll")) return 1;
    reply = transfer(memory, 0u, {0x01, 0x44, 0x00, 0x01, 0x03, 0x00, 0x00, 0x00, 0x00});
    if (!expect(reply[1] == 0xF3u && reply[2] == 0x5Au, "configuration mode byte")) return 1;
    reply = transfer(memory, 0u, {0x01, 0x45, 0x00, 0x5A, 0x5A, 0x5A, 0x5A, 0x5A, 0x5A});
    if (!expect(reply[3] == 0x03u && reply[5] == 0x01u, "model query: DualShock 2 with analog LED on")) return 1;
    reply = transfer(memory, 0u, {0x01, 0x43, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00});
    buttons = 0xFFFFu;
    reply = transfer(memory, 0u, {0x01, 0x42, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00});
    if (!expect(reply == std::vector<uint8_t>{0xFF, 0x73, 0x5A, 0xFF, 0xFF, 0x11, 0x22, 0x33, 0x44}, "analog poll reply")) return 1;

    // Other ports: no device.
    reply = transfer(memory, 1u, {0x01, 0x42, 0x00, 0x00, 0x00});
    if (!expect(reply[1] == 0xFFu && memory.read32(kRecv1) == 0x1D100u, "port 1 reports no device")) return 1;

    // DMA out: replies land in RAM instead of the FIFO.
    sio2.armDmaFromDevice(0x1000u, 9u);
    (void)transfer(memory, 0u, {0x01, 0x42, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00});
    uint8_t dma[9]{};
    if (!expect(memory.readRam(0x1000u, dma, sizeof(dma)) && dma[1] == 0x73u && dma[2] == 0x5Au, "DMA out receives the reply")) return 1;

    std::cout << "iop sio2 tests passed\n";
    return 0;
}
