#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <vector>

namespace ps2x::iop::detail
{
    class IopMemory;

    // DualShock 2 on a SIO2 port: digital, analog and pressure modes, and the configuration
    // commands padman and Sony's dbc driver use to switch between them.
    class IopDualShock2
    {
    public:
        // 32-byte pad state as libpad reports it: [2..3] buttons (active low), [4..7] rx, ry, lx, ly.
        using StateReader = std::function<bool(uint8_t *state, size_t size)>;

        void reset();
        // One full-duplex exchange: the reply has the size of the command.
        [[nodiscard]] std::vector<uint8_t> exchange(const std::vector<uint8_t> &command, const StateReader &readState);

    private:
        [[nodiscard]] uint8_t modeByte() const;

        bool m_analog = false;
        bool m_pressure = false;
        bool m_pressureRequested = false;
        bool m_config = false;
    };

    // SIO2 controller (IOP 0x1F808200-0x1F808284): transfers queued in SEND3, data through the
    // DATAIN/DATAOUT FIFOs or DMA channels 11/12, completion signalled by interrupt 17.
    // Port 0 holds a DualShock 2; the other ports (pad 2, memory cards) report no device.
    class IopSio2
    {
    public:
        static constexpr uint32_t Base = 0x1F808200u;
        static constexpr uint32_t End = 0x1F808284u;
        static constexpr int Irq = 17;
        static constexpr uint32_t DmaInChannel = 11u;
        static constexpr uint32_t DmaOutChannel = 12u;

        explicit IopSio2(IopMemory &memory) noexcept;

        void reset();
        void setPadStateReader(IopDualShock2::StateReader reader) { m_readPadState = std::move(reader); }

        [[nodiscard]] static bool handles(uint32_t address) noexcept { return address >= Base && address <= End; }
        [[nodiscard]] uint32_t read32(uint32_t address);
        void write32(uint32_t address, uint32_t value);

        // dmacman: channel 11 feeds DATAIN from RAM now; channel 12 receives the next replies.
        void dmaToDevice(uint32_t address, uint32_t bytes);
        void armDmaFromDevice(uint32_t address, uint32_t bytes);

    private:
        void runTransfer();

        IopMemory &m_memory;
        IopDualShock2 m_pad;
        IopDualShock2::StateReader m_readPadState;
        std::array<uint32_t, 16> m_send3{};
        std::array<uint32_t, 8> m_send12{};
        std::deque<uint8_t> m_in;
        std::deque<uint8_t> m_out;
        uint32_t m_ctrl = 0u;
        uint32_t m_recv1 = 0x1D100u;
        uint32_t m_istat = 0u;
        uint32_t m_outDmaAddress = 0u;
        uint32_t m_outDmaBytes = 0u;
    };
}
