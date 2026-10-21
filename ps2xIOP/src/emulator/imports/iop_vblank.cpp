#include "iop_vblank.h"

#include "../core/iop_cpu.h"
#include "../iop_emulator_const.h"
#include "../core/iop_kernel.h"
#include "../services/iop_rpc.h"

#include <algorithm>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr int32_t kErrorFoundHandler = -104;
        constexpr int32_t kErrorNotFoundHandler = -105;
        constexpr uint32_t kHandlerBudget = 100000u;

        uint64_t nextEdgeAfter(uint64_t cycle, uint32_t edge) noexcept
        {
            const uint64_t phase = edge == 0u ? 0u : kVblankEndPhaseCycles;
            uint64_t next = cycle - (cycle % kVblankPeriodCycles) + phase;
            if (next <= cycle)
                next += kVblankPeriodCycles;
            return next;
        }
    }

    IopVblank::IopVblank(IopKernel &kernel) noexcept
        : m_kernel(kernel)
    {
        reset();
    }

    void IopVblank::reset() noexcept
    {
        m_handlers.clear();
        m_nextEdge[0] = nextEdgeAfter(0u, 0u);
        m_nextEdge[1] = nextEdgeAfter(0u, 1u);
        m_servicing = false;
    }

    bool IopVblank::dispatchImport(uint16_t ordinal, IopCpuState &cpu, uint64_t currentCycle)
    {
        switch (ordinal)
        {
        case 4: // WaitVblankStart
        case 5: // WaitVblankEnd
        case 6: // WaitVblank
        case 7: // WaitNonVblank
        {
            const bool waitForEnd = ordinal == 5u || ordinal == 7u;
            m_kernel.delayCurrentUntil(nextEdgeAfter(currentCycle, waitForEnd ? 1u : 0u), cpu);
            cpu.gpr[2] = 0u;
            return true;
        }
        case 8: // RegisterVblankHandler(edge, priority, handler, arg)
        {
            const uint32_t edge = cpu.gpr[4] != 0u ? 1u : 0u;
            const uint32_t function = cpu.gpr[6];
            const bool exists = std::any_of(m_handlers.begin(), m_handlers.end(), [&](const Handler &handler)
                                            { return handler.edge == edge && handler.function == function; });
            if (exists || function == 0u)
            {
                cpu.gpr[2] = static_cast<uint32_t>(kErrorFoundHandler);
                return true;
            }
            if (std::none_of(m_handlers.begin(), m_handlers.end(), [&](const Handler &handler)
                             { return handler.edge == edge; }))
                m_nextEdge[edge] = nextEdgeAfter(currentCycle, edge);
            m_handlers.push_back({edge, static_cast<int32_t>(cpu.gpr[5]), function, cpu.gpr[7], cpu.gpr[28]});
            // Lower priority values run first; registration order breaks ties.
            std::stable_sort(m_handlers.begin(), m_handlers.end(), [](const Handler &a, const Handler &b)
                             { return a.priority < b.priority; });
            cpu.gpr[2] = 0u;
            return true;
        }
        case 9: // ReleaseVblankHandler(edge, handler)
        {
            const uint32_t edge = cpu.gpr[4] != 0u ? 1u : 0u;
            const auto handler = std::find_if(m_handlers.begin(), m_handlers.end(), [&](const Handler &candidate)
                                              { return candidate.edge == edge && candidate.function == cpu.gpr[5]; });
            if (handler == m_handlers.end())
            {
                cpu.gpr[2] = static_cast<uint32_t>(kErrorNotFoundHandler);
                return true;
            }
            m_handlers.erase(handler);
            cpu.gpr[2] = 0u;
            return true;
        }
        default:
            return false;
        }
    }

    void IopVblank::serviceDue(uint64_t currentCycle, IopGuestExecutor &executor)
    {
        if (m_servicing || m_handlers.empty())
            return;
        m_servicing = true;
        struct ServiceGuard
        {
            bool &flag;
            ~ServiceGuard() { flag = false; }
        } guard{m_servicing};

        for (uint32_t edge = 0u; edge < 2u; ++edge)
        {
            if (m_nextEdge[edge] > currentCycle)
                continue;
            // One delivery per edge, even when the IOP ran several fields at once.
            m_nextEdge[edge] = nextEdgeAfter(currentCycle, edge);
            const std::vector<Handler> handlers = m_handlers;
            for (const Handler &handler : handlers)
            {
                if (handler.edge != edge)
                    continue;
                const uint32_t keep = executor.executeGuestFunctionWithBudget(handler.function, handler.argument, 0u, 0u,
                                                                              0u, handler.gp, kHandlerBudget);
                if (keep == 0u)
                {
                    const auto released = std::find_if(m_handlers.begin(), m_handlers.end(), [&](const Handler &candidate)
                                                       { return candidate.edge == edge && candidate.function == handler.function; });
                    if (released != m_handlers.end())
                        m_handlers.erase(released);
                }
            }
        }
    }

    uint64_t IopVblank::nextEventCycle(uint64_t fallback) const noexcept
    {
        uint64_t next = fallback;
        for (const Handler &handler : m_handlers)
            next = std::min(next, m_nextEdge[handler.edge]);
        return next;
    }
}
