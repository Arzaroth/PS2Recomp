#include "iop_sio2.h"

#include "iop_memory.h"

#include <algorithm>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint32_t kRecv1Connected = 0x1100u;
        constexpr uint32_t kRecv1NoDevice = 0x1D100u;
        constexpr uint8_t kPadAddress = 0x01u;

        // Pressure byte order after the sticks: right, left, up, down, triangle, circle, cross,
        // square, L1, R1, L2, R2 (button bits in the active-low 16-bit word).
        constexpr std::array<uint16_t, 12> kPressureButtons{
            0x0020u, 0x0080u, 0x0010u, 0x0040u, 0x1000u, 0x2000u,
            0x4000u, 0x8000u, 0x0400u, 0x0800u, 0x0100u, 0x0200u,
        };
    }

    void IopDualShock2::reset()
    {
        m_analog = false;
        m_pressure = false;
        m_pressureRequested = false;
        m_config = false;
    }

    uint8_t IopDualShock2::modeByte() const
    {
        if (m_config)
            return 0xF3u;
        if (!m_analog)
            return 0x41u;
        return m_pressure ? 0x79u : 0x73u;
    }

    std::vector<uint8_t> IopDualShock2::exchange(const std::vector<uint8_t> &command, const StateReader &readState)
    {
        std::vector<uint8_t> reply(command.size(), 0x00u);
        if (command.size() < 2u || command[0] != kPadAddress)
        {
            std::fill(reply.begin(), reply.end(), 0xFFu);
            return reply;
        }
        const uint8_t code = command[1];
        const auto arg = [&](size_t index) -> uint8_t
        { return index < command.size() ? command[index] : 0u; };
        std::vector<uint8_t> data;

        const bool polling = code == 0x42u || (code == 0x43u && !m_config);
        const uint8_t mode = modeByte();
        if (polling)
        {
            std::array<uint8_t, 32> state{};
            state[2] = state[3] = 0xFFu;
            state[4] = state[5] = state[6] = state[7] = 0x80u;
            if (readState)
                (void)readState(state.data(), state.size());
            data = {state[2], state[3]};
            if (m_analog)
                data.insert(data.end(), {state[4], state[5], state[6], state[7]});
            if (m_analog && m_pressure)
            {
                const uint16_t buttons = static_cast<uint16_t>(state[2] | (state[3] << 8));
                for (const uint16_t mask : kPressureButtons)
                    data.push_back((buttons & mask) == 0u ? 0xFFu : 0x00u);
            }
            if (code == 0x43u && arg(3) == 1u)
                m_config = true;
        }
        else if (m_config)
        {
            switch (code)
            {
            case 0x43u: // exit configuration
                if (arg(3) == 0u)
                {
                    m_config = false;
                    m_pressure = m_analog && m_pressureRequested;
                }
                data.assign(6u, 0x00u);
                break;
            case 0x44u: // set main mode
                m_analog = arg(3) == 1u;
                if (!m_analog)
                    m_pressureRequested = false;
                data.assign(6u, 0x00u);
                break;
            case 0x45u: // query model: DualShock 2, current LED state
                data = {0x03u, 0x02u, static_cast<uint8_t>(m_analog ? 0x01u : 0x00u), 0x02u, 0x01u, 0x00u};
                break;
            case 0x46u:
                data = arg(3) == 0u ? std::vector<uint8_t>{0x00u, 0x00u, 0x01u, 0x02u, 0x00u, 0x0Au}
                                    : std::vector<uint8_t>{0x00u, 0x00u, 0x01u, 0x01u, 0x01u, 0x14u};
                break;
            case 0x47u:
                data = {0x00u, 0x00u, 0x02u, 0x00u, 0x01u, 0x00u};
                break;
            case 0x4Cu:
                data = arg(3) == 0u ? std::vector<uint8_t>{0x00u, 0x00u, 0x00u, 0x04u, 0x00u, 0x00u}
                                    : std::vector<uint8_t>{0x00u, 0x00u, 0x00u, 0x07u, 0x00u, 0x00u};
                break;
            case 0x4Du: // actuator mapping: report the previous (unmapped) one
                data.assign(6u, 0xFFu);
                break;
            case 0x4Fu: // pressure-sensitive response mask
                m_pressureRequested = (arg(3) | arg(4) | arg(5)) != 0u;
                data = {0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x5Au};
                break;
            case 0x41u:
                data = {0xFFu, 0xFFu, 0x03u, 0x00u, 0x00u, 0x5Au};
                break;
            case 0x40u:
                data = {0x00u, 0x00u, 0x02u, 0x00u, 0x00u, 0x5Au};
                break;
            default:
                data.assign(6u, 0x00u);
                break;
            }
        }
        else
        {
            data.assign(2u, 0xFFu); // configuration-only command outside configuration mode
        }

        reply[0] = 0xFFu;
        if (reply.size() > 1u)
            reply[1] = mode;
        if (reply.size() > 2u)
            reply[2] = 0x5Au;
        for (size_t i = 0; i < data.size() && i + 3u < reply.size(); ++i)
            reply[i + 3u] = data[i];
        return reply;
    }

    IopSio2::IopSio2(IopMemory &memory) noexcept
        : m_memory(memory)
    {
    }

    void IopSio2::reset()
    {
        m_pad.reset();
        m_send3.fill(0u);
        m_send12.fill(0u);
        m_in.clear();
        m_out.clear();
        m_ctrl = 0u;
        m_recv1 = kRecv1NoDevice;
        m_istat = 0u;
        m_outDmaAddress = 0u;
        m_outDmaBytes = 0u;
    }

    uint32_t IopSio2::read32(uint32_t address)
    {
        if (address < 0x1F808240u)
            return m_send3[(address - Base) / 4u];
        if (address < 0x1F808260u)
            return m_send12[(address - 0x1F808240u) / 4u];
        switch (address)
        {
        case 0x1F808264u: // DATAOUT
        {
            if (m_out.empty())
                return 0u;
            const uint8_t value = m_out.front();
            m_out.pop_front();
            return value;
        }
        case 0x1F808268u:
            return m_ctrl;
        case 0x1F80826Cu:
            return m_recv1;
        case 0x1F808270u:
            return 0xFu;
        case 0x1F808280u:
            return m_istat;
        default:
            return 0u;
        }
    }

    void IopSio2::write32(uint32_t address, uint32_t value)
    {
        if (address < 0x1F808240u)
        {
            m_send3[(address - Base) / 4u] = value;
            return;
        }
        if (address < 0x1F808260u)
        {
            m_send12[(address - 0x1F808240u) / 4u] = value;
            return;
        }
        switch (address)
        {
        case 0x1F808260u: // DATAIN
            m_in.push_back(static_cast<uint8_t>(value));
            return;
        case 0x1F808268u: // CTRL: bit 0 starts the queued transfers; otherwise bits 2-3 reset the FIFOs
            m_ctrl = value & ~1u;
            if ((value & 1u) != 0u)
            {
                runTransfer();
            }
            else if ((value & 0xCu) != 0u)
            {
                m_in.clear();
                m_out.clear();
            }
            return;
        case 0x1F808280u: // ISTAT: write 1 to acknowledge
            m_istat &= ~value;
            return;
        default:
            return;
        }
    }

    void IopSio2::dmaToDevice(uint32_t address, uint32_t bytes)
    {
        std::vector<uint8_t> data(bytes);
        if (bytes != 0u && m_memory.readRam(address, data.data(), data.size()))
            m_in.insert(m_in.end(), data.begin(), data.end());
    }

    void IopSio2::armDmaFromDevice(uint32_t address, uint32_t bytes)
    {
        m_outDmaAddress = address;
        m_outDmaBytes = bytes;
    }

    void IopSio2::runTransfer()
    {
        std::vector<uint8_t> produced;
        for (const uint32_t control : m_send3)
        {
            if (control == 0u)
                break;
            const uint32_t port = control & 3u;
            const uint32_t inSize = (control >> 8) & 0x1FFu;
            const uint32_t outSize = (control >> 18) & 0x1FFu;
            std::vector<uint8_t> command;
            for (uint32_t i = 0; i < inSize && !m_in.empty(); ++i)
            {
                command.push_back(m_in.front());
                m_in.pop_front();
            }
            command.resize(std::max(inSize, outSize), 0x00u);

            std::vector<uint8_t> reply;
            if (port == 0u)
            {
                reply = m_pad.exchange(command, m_readPadState);
                m_recv1 = kRecv1Connected;
            }
            else
            {
                reply.assign(command.size(), 0xFFu);
                m_recv1 = kRecv1NoDevice;
            }
            reply.resize(outSize, 0x00u);
            produced.insert(produced.end(), reply.begin(), reply.end());
        }

        if (m_outDmaBytes != 0u)
        {
            const uint32_t count = std::min<uint32_t>(m_outDmaBytes, static_cast<uint32_t>(produced.size()));
            (void)m_memory.writeRam(m_outDmaAddress, produced.data(), count);
            produced.erase(produced.begin(), produced.begin() + count);
            m_outDmaAddress = 0u;
            m_outDmaBytes = 0u;
        }
        m_out.insert(m_out.end(), produced.begin(), produced.end());

        m_istat |= 1u;
        m_memory.requestInterrupt(Irq, 64u);
    }
}
