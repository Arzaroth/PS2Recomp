#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <utility>

// EE cycles the GS spends drawing the primitives sent to it, from their scissored area. Only what decides
// the GS's speed at the scale of a handshake is modelled: pixels filled and a fixed cost per primitive.
class GsDrawEstimate
{
public:
    struct Rates
    {
        // EE cycles per 1024 pixels: the GS fills about 4 textured, blended pixels per GS cycle (2 EE cycles).
        uint64_t cyclesPer1024Pixels = 512u;
        uint64_t cyclesPerPrimitive = 32u;
    };

    void setRates(Rates rates) { m_rates = rates; }

    void setPrim(uint64_t value)
    {
        m_prim = static_cast<uint32_t>(value & 0x7u);
        m_context = static_cast<uint32_t>((value >> 9) & 0x1u);
        m_count = 0u;
    }

    void setXyOffset(uint32_t context, uint64_t value)
    {
        m_offsetX[context & 1u] = static_cast<int32_t>(value & 0xFFFFu);
        m_offsetY[context & 1u] = static_cast<int32_t>((value >> 32) & 0xFFFFu);
    }

    void setScissor(uint32_t context, uint64_t value)
    {
        Rect &scissor = m_scissor[context & 1u];
        scissor.x0 = static_cast<int32_t>(value & 0x7FFu);
        scissor.x1 = static_cast<int32_t>((value >> 16) & 0x7FFu);
        scissor.y0 = static_cast<int32_t>((value >> 32) & 0x7FFu);
        scissor.y1 = static_cast<int32_t>((value >> 48) & 0x7FFu);
    }

    // x and y are the 12.4 fixed-point XYZ coordinates; kick is false for XYZ3/XYZF3 and ADC vertices.
    void vertex(uint32_t x, uint32_t y, bool kick)
    {
        m_x[0] = m_x[1];
        m_y[0] = m_y[1];
        m_x[1] = m_x[2];
        m_y[1] = m_y[2];
        m_x[2] = static_cast<int32_t>(x & 0xFFFFu) - m_offsetX[m_context];
        m_y[2] = static_cast<int32_t>(y & 0xFFFFu) - m_offsetY[m_context];
        if (m_count == 0u)
        {
            m_fanX = m_x[2];
            m_fanY = m_y[2];
        }
        ++m_count;
        if (kick)
            draw();
    }

    // GS register write as the GS sees it (PACKED data already converted to the register's format).
    void writeRegister(uint8_t address, uint64_t value)
    {
        switch (address)
        {
        case 0x00u:
            setPrim(value);
            break;
        case 0x04u:
        case 0x05u:
        case 0x0Cu:
        case 0x0Du:
            vertex(static_cast<uint32_t>(value), static_cast<uint32_t>(value >> 16), address == 0x04u || address == 0x05u);
            break;
        case 0x18u:
        case 0x19u:
            setXyOffset(address - 0x18u, value);
            break;
        case 0x40u:
        case 0x41u:
            setScissor(address - 0x40u, value);
            break;
        default:
            break;
        }
    }

    // A host-to-local transfer moves a quadword per GS cycle.
    void addImageBytes(uint64_t bytes)
    {
        m_cycles += bytes / 8u;
        m_total += bytes / 8u;
    }

    // Since the previous SIGNAL/FINISH/LABEL, and since the start.
    uint64_t takeCycles() { return std::exchange(m_cycles, 0u); }
    uint64_t totalCycles() const { return m_total; }

private:
    struct Rect
    {
        int32_t x0 = 0, x1 = 2047, y0 = 0, y1 = 2047;
    };

    // Pixels of the 12.4 bounding box inside the scissor.
    uint64_t clippedBoxPixels(int32_t minX, int32_t maxX, int32_t minY, int32_t maxY) const
    {
        const Rect &scissor = m_scissor[m_context];
        const int32_t x0 = std::max(minX >> 4, scissor.x0), x1 = std::min(maxX >> 4, scissor.x1);
        const int32_t y0 = std::max(minY >> 4, scissor.y0), y1 = std::min(maxY >> 4, scissor.y1);
        if (x1 < x0 || y1 < y0)
            return 0u;
        return static_cast<uint64_t>(x1 - x0 + 1) * static_cast<uint64_t>(y1 - y0 + 1);
    }

    void addPrimitive(uint64_t pixels)
    {
        const uint64_t cycles = m_rates.cyclesPerPrimitive + pixels * m_rates.cyclesPer1024Pixels / 1024u;
        m_cycles += cycles;
        m_total += cycles;
    }

    void draw()
    {
        switch (m_prim)
        {
        case 0u:
            addPrimitive(1u);
            m_count = 0u;
            break;
        case 1u:
        case 2u:
            if (m_count >= 2u)
            {
                addPrimitive(static_cast<uint64_t>(std::max(std::abs(m_x[2] - m_x[1]), std::abs(m_y[2] - m_y[1])) >> 4) + 1u);
                if (m_prim == 1u)
                    m_count = 0u;
            }
            break;
        case 3u:
        case 4u:
        case 5u:
            if (m_count >= 3u)
            {
                const int32_t ax = m_prim == 5u ? m_fanX : m_x[0];
                const int32_t ay = m_prim == 5u ? m_fanY : m_y[0];
                const int64_t cross = static_cast<int64_t>(m_x[1] - ax) * (m_y[2] - ay) -
                                      static_cast<int64_t>(m_x[2] - ax) * (m_y[1] - ay);
                const uint64_t area = static_cast<uint64_t>(std::llabs(cross)) / 512u;
                const uint64_t box = clippedBoxPixels(std::min({ax, m_x[1], m_x[2]}), std::max({ax, m_x[1], m_x[2]}),
                                                      std::min({ay, m_y[1], m_y[2]}), std::max({ay, m_y[1], m_y[2]}));
                addPrimitive(std::min(area, box));
                if (m_prim == 3u)
                    m_count = 0u;
            }
            break;
        case 6u:
            if (m_count >= 2u)
            {
                addPrimitive(clippedBoxPixels(std::min(m_x[1], m_x[2]), std::max(m_x[1], m_x[2]) - 1, std::min(m_y[1], m_y[2]),
                                              std::max(m_y[1], m_y[2]) - 1));
                m_count = 0u;
            }
            break;
        default:
            break;
        }
    }

    Rates m_rates{};
    uint32_t m_prim = 0u;
    uint32_t m_context = 0u;
    uint32_t m_count = 0u;
    int32_t m_x[3]{}, m_y[3]{};
    int32_t m_fanX = 0, m_fanY = 0;
    int32_t m_offsetX[2]{}, m_offsetY[2]{};
    Rect m_scissor[2]{};
    uint64_t m_cycles = 0u;
    uint64_t m_total = 0u;
};
