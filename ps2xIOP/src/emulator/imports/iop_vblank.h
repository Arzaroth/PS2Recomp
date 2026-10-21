#pragma once

#include <cstdint>
#include <vector>

namespace ps2x::iop::detail
{
    struct IopCpuState;
    class IopGuestExecutor;
    class IopKernel;

    class IopVblank
    {
    public:
        explicit IopVblank(IopKernel &kernel) noexcept;

        void reset() noexcept;
        [[nodiscard]] bool dispatchImport(uint16_t ordinal, IopCpuState &cpu, uint64_t currentCycle);
        // Runs the handlers of the VBlank start / end edges due by currentCycle, like the VBlank
        // interrupt: in priority order, a handler returning 0 is released.
        void serviceDue(uint64_t currentCycle, IopGuestExecutor &executor);
        [[nodiscard]] uint64_t nextEventCycle(uint64_t fallback) const noexcept;

    private:
        struct Handler
        {
            uint32_t edge = 0u;
            int32_t priority = 0;
            uint32_t function = 0u;
            uint32_t argument = 0u;
            uint32_t gp = 0u;
        };

        IopKernel &m_kernel;
        std::vector<Handler> m_handlers;
        // Next edge of each kind: 0 = VBlank start, 1 = VBlank end.
        uint64_t m_nextEdge[2]{};
        bool m_servicing = false;
    };
}
