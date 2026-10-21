#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace GSMem
{
    class TexturePageCache
    {
    public:
        static constexpr uint32_t kPageSize = 8192u;
        // Several pages: with one, a primitive whose texels straddle a page boundary (bilinear
        // taps, rotated triangles) recopied 8 KiB per texel.
        static constexpr uint32_t kPageCount = 8u;

        void Invalidate() noexcept
        {
            m_pageBases.fill(UINT32_MAX);
        }

        // byteAddress is the wrapped, swizzled VRAM address. The returned
        // pointer is valid only until the next miss or invalidation.
        const uint8_t* Resolve(const uint8_t* vram, uint32_t byteAddress) noexcept
        {
            const uint32_t pageBase = byteAddress & ~(kPageSize - 1u);
            const uint32_t offset = byteAddress & (kPageSize - 1u);
            if (m_pageBases[m_lastHit] == pageBase)
            {
                return m_pages[m_lastHit].data() + offset;
            }
            uint32_t victim = 0u;
            for (uint32_t i = 0; i < kPageCount; ++i)
            {
                if (m_pageBases[i] == pageBase)
                {
                    m_lastUse[i] = ++m_useClock;
                    m_lastHit = i;
                    return m_pages[i].data() + offset;
                }
                if (m_lastUse[i] < m_lastUse[victim])
                {
                    victim = i;
                }
            }
            std::memcpy(m_pages[victim].data(), vram + pageBase, kPageSize);
            m_pageBases[victim] = pageBase;
            m_lastUse[victim] = ++m_useClock;
            m_lastHit = victim;
            return m_pages[victim].data() + offset;
        }

    private:
        alignas(64) std::array<std::array<uint8_t, kPageSize>, kPageCount> m_pages{};
        std::array<uint32_t, kPageCount> m_pageBases = makeEmptyBases();
        std::array<uint64_t, kPageCount> m_lastUse{};
        uint64_t m_useClock = 0u;
        uint32_t m_lastHit = 0u;

        static constexpr std::array<uint32_t, kPageCount> makeEmptyBases() noexcept
        {
            std::array<uint32_t, kPageCount> bases{};
            bases.fill(UINT32_MAX);
            return bases;
        }
    };
}
