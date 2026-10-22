#include "iop_emulator.h"
#include "imports/iop_cdvd.h"
#include "core/iop_cpu.h"
#include "imports/iop_heaplib.h"
#include "imports/iop_imports.h"
#include "imports/iop_intrman.h"
#include "imports/iop_ioman.h"
#include "core/iop_kernel.h"
#include "imports/iop_loadcore.h"
#include "core/iop_memory.h"
#include "core/iop_sio2.h"
#include "core/iop_spu2.h"
#include "services/iop_module_loader.h"
#include "services/iop_rpc.h"
#include "imports/iop_stdio.h"
#include "imports/iop_sysclib.h"
#include "imports/iop_sysmem.h"
#include "imports/iop_timrman.h"
#include "imports/iop_vblank.h"
#include "iop_emulator_const.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ps2x::iop::detail
{
    namespace
    {
        // PS2X_IOP_TRACE_PC=0x62914,0x6291c: log a0-a3 and ra each time the IOP reaches one of these addresses.
        const std::unordered_set<uint32_t> *tracedPcs()
        {
            static const std::unordered_set<uint32_t> *pcs = []() -> const std::unordered_set<uint32_t> *
            {
                const char *spec = std::getenv("PS2X_IOP_TRACE_PC");
                if (!spec || !*spec)
                    return nullptr;
                auto *set = new std::unordered_set<uint32_t>();
                std::string text(spec);
                for (size_t start = 0; start < text.size();)
                {
                    size_t end = text.find(',', start);
                    if (end == std::string::npos)
                        end = text.size();
                    set->insert(static_cast<uint32_t>(std::strtoul(text.substr(start, end - start).c_str(), nullptr, 0)));
                    start = end + 1;
                }
                return set;
            }();
            return pcs;
        }

        // PS2X_IOP_COUNT_RANGE=0x56b00-0x63000: count executed instructions per pc in that range and write
        // "pc count" lines to PS2X_IOP_COUNT_OUT (default iop_pc_counts.txt) every 2^24 counted instructions.
        struct PcCounter
        {
            uint32_t low = 0;
            uint32_t high = 0;
            std::vector<uint64_t> counts;
            uint64_t total = 0;

            void write() const
            {
                const char *out = std::getenv("PS2X_IOP_COUNT_OUT");
                if (FILE *f = std::fopen(out && *out ? out : "iop_pc_counts.txt", "w"))
                {
                    for (size_t i = 0; i < counts.size(); ++i)
                        if (counts[i] != 0u)
                            std::fprintf(f, "0x%x %llu\n", low + static_cast<uint32_t>(i) * 4u,
                                         static_cast<unsigned long long>(counts[i]));
                    std::fclose(f);
                }
            }
        };

        PcCounter *pcCounter()
        {
            static PcCounter *counter = []() -> PcCounter *
            {
                const char *spec = std::getenv("PS2X_IOP_COUNT_RANGE");
                if (!spec || !*spec)
                    return nullptr;
                char *end = nullptr;
                auto *result = new PcCounter();
                result->low = static_cast<uint32_t>(std::strtoul(spec, &end, 0));
                result->high = static_cast<uint32_t>(std::strtoul(end && *end == '-' ? end + 1 : "0", nullptr, 0));
                if (result->high <= result->low)
                    return nullptr;
                result->counts.assign((result->high - result->low) / 4u, 0u);
                return result;
            }();
            return counter;
        }

        constexpr uint32_t kRamSize = IopMemory::RamSize;
        constexpr uint32_t kKernelHeapBase = IopMemory::HeapBase;
        constexpr uint32_t kKernelHeapLimit = IopMemory::HeapLimit;
        constexpr uint32_t kCallStackBase = kKernelHeapLimit;
        constexpr uint32_t kCallStackLimit = 0x001FFF00u;
        constexpr uint32_t kCallStackSize = 0x2000u;
        constexpr uint32_t kCallStackCapacity = (kCallStackLimit - kCallStackBase) / kCallStackSize;
        constexpr uint64_t kCdvdCompletionCycles = 128u;
        constexpr uint32_t kMaxSifCommandPacket = 112u;
        constexpr uint64_t kSifDmaCompletionCycles = 128u;
        constexpr uint32_t kSifCommandHandlerBudget = 100000u;

        uint32_t physicalAddress(uint32_t address)
        {
            return IopMemory::physicalAddress(address);
        }

        int32_t sign16(uint32_t value)
        {
            return static_cast<int16_t>(value & 0xFFFFu);
        }

        bool iequals(std::string_view lhs, std::string_view rhs)
        {
            if (lhs.size() != rhs.size())
                return false;
            for (size_t i = 0; i < lhs.size(); ++i)
            {
                if (std::tolower(static_cast<unsigned char>(lhs[i])) !=
                    std::tolower(static_cast<unsigned char>(rhs[i])))
                    return false;
            }
            return true;
        }

    }

    class IopEmulator::Impl final : public IopGuestExecutor
    {
    public:
        using CpuState = IopCpuState;

        struct Module
        {
            int id = 0;
            std::string path;
            std::string name;
            uint32_t base = 0;
            uint32_t size = 0;
            uint32_t entry = 0;
            uint32_t gp = 0;
            bool resident = false;
        };

        struct GuestCallback
        {
            uint32_t function = 0;
            uint32_t gp = 0;
        };

        struct ScheduledGuestCallback
        {
            uint32_t function = 0u;
            uint32_t gp = 0u;
            uint32_t argument = 0u;
        };

        // EE -> IOP SIF command waiting for its IOP handler (sceSifAddCmdHandler).
        struct PendingSifCommand
        {
            IopRpcBridge::CmdHandler handler;
            std::array<uint8_t, kMaxSifCommandPacket> packet{};
            uint32_t size = 0u;
        };

        explicit Impl(IopHost &hostRef)
            : host(hostRef),
              sysmem(host, memory),
              kernel(memory),
              cdvd(host, memory, kernel),
              vblank(kernel),
              rpc(host, memory, kernel),
              sysclib(memory),
              stdio(host, memory),
              heaplib(memory),
              intrman(memory),
              timrman(),
              ioman(memory),
              cpuCore(memory),
              imports(memory),
              loadcore(memory, imports),
              sio2(memory),
              spu2(memory)
        {
            memory.setSio2(&sio2);
            memory.setSpu2(&spu2);
            spu2.setClock(&totalCycles);
            spu2.setInterruptSink([this](int irq)
                                  { pendingDmaInterrupts[irq] = totalCycles; });
            spu2.setOutputSink([this](const int16_t *frames, size_t count)
                               { host.submitAudio(frames, count); });
            sio2.setPadStateReader([this](uint8_t *state, size_t size)
                                   { return host.readPadState(0, state, size); });
            reset();
        }

        void reset()
        {
            memory.reset();
            kernel.reset();
            modules.clear();
            imports.reset();
            rpc.reset();
            cdvd.reset();
            intrman.reset();
            timrman.reset();
            vblank.reset();
            ioman.reset();
            pendingDmaInterrupts.clear();
            pendingGuestCallbacks.clear();
            pendingSifCommands.clear();
            sifCommandBuffer = 0u;
            sio2.reset();
            spu2.reset();
            dmaRequests = {};
            nextModuleId = 1;
            moduleCursor = kModuleLoadBase;
            totalCycles = 0;
            totalInstructions = 0;
            eeCycleCarry = 0;
            activeCpu = nullptr;
            lastError.clear();
            servicingDmaInterrupts = false;
            servicingGuestCallbacks = false;
            callDepth = 0u;
            secrMcCommandHandler = {};
            secrMcDevIdHandler = {};
            checkKelfPathCallback = {};
        }

        uint8_t read8(uint32_t address) const
        {
            return memory.read8(address);
        }

        uint16_t read16(uint32_t address) const
        {
            return memory.read16(address);
        }

        uint32_t read32(uint32_t address) const
        {
            return memory.read32(address);
        }

        void write8(uint32_t address, uint8_t value)
        {
            memory.write8(address, value);
            schedulePendingDma();
        }

        void write16(uint32_t address, uint16_t value)
        {
            memory.write16(address, value);
            schedulePendingDma();
        }

        void write32(uint32_t address, uint32_t value)
        {
            memory.write32(address, value);
            schedulePendingDma();
        }

        void schedulePendingDma()
        {
            if (const auto dma = memory.takeDmaStart())
                pendingDmaInterrupts[dma->irq] = totalCycles + dma->delayCycles;
        }

        bool readRam(uint32_t address, void *destination, size_t size) const
        {
            return memory.readRam(address, destination, size);
        }

        bool writeRam(uint32_t address, const void *source, size_t size)
        {
            return memory.writeRam(address, source, size);
        }

        bool zeroRam(uint32_t address, size_t size)
        {
            return memory.zeroRam(address, size);
        }

        bool isHardwareAddress(uint32_t phys) const
        {
            return memory.isHardwareAddress(phys);
        }

        uint32_t allocate(uint32_t size, uint32_t alignment = 16u, std::optional<uint32_t> fixed = std::nullopt)
        {
            return memory.allocate(size, alignment, fixed);
        }

        bool freeAllocation(uint32_t address)
        {
            return memory.freeAllocation(address);
        }

        void log(LogLevel level, std::string_view text)
        {
            host.log(level, text);
        }

        bool checkInterrupt(CpuState &cpu)
        {
            const uint32_t status = cpu.cop0[12];
            if ((status & 1u) == 0u)
                return false;
            if ((status & 0x2u) != 0u)
                return false;
            const bool pending = memory.interruptControl() != 0u && (memory.interruptStatus() & memory.interruptMask()) != 0u;
            if (!pending)
                return false;
            cpu.cop0[13] |= 0x400u;
            cpuCore.raiseException(cpu, 0u, cpu.pc, false);
            return true;
        }

        enum class ImportDisposition
        {
            Handled,
            JumpToGuest,
            Missing,
        };

        ImportDisposition dispatchImport(const IopImportCall &call, CpuState &cpu)
        {
            const uint32_t a0 = cpu.gpr[4];
            auto setV0 = [&](uint32_t value)
            {
                cpu.gpr[2] = value;
            };

            if (iequals(call.library, "sysmem") && sysmem.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;

            if (iequals(call.library, "cdvdman") && cdvd.dispatchImport(call.ordinal, cpu))
            {
                if (const auto callback = cdvd.takeCompletionCallback())
                {
                    pendingGuestCallbacks.emplace(
                        totalCycles + kCdvdCompletionCycles,
                        ScheduledGuestCallback{
                            callback->address,
                            callback->gp,
                            callback->reason,
                        });
                }
                return ImportDisposition::Handled;
            }

            if (iequals(call.library, "loadcore") && loadcore.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;

            if (iequals(call.library, "thbase") || iequals(call.library, "threadman"))
            {
                return kernel.dispatchThreadImport(call.ordinal, cpu, totalCycles)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "thsemap"))
            {
                return kernel.dispatchSemaphoreImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "thevent"))
            {
                return kernel.dispatchEventImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "sifcmd"))
            {
                return rpc.dispatchSifCmdImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "intrman") && intrman.dispatchImport(call.ordinal, cpu, *this))
                return ImportDisposition::Handled;
            if (iequals(call.library, "secrman"))
            {
                switch (call.ordinal)
                {
                case 4: // SecrSetMcCommandHandler
                    secrMcCommandHandler = {a0, cpu.gpr[28]};
                    setV0(0);
                    return ImportDisposition::Handled;
                case 5: // SecrSetMcDevIDHandler
                    secrMcDevIdHandler = {a0, cpu.gpr[28]};
                    setV0(0);
                    return ImportDisposition::Handled;
                default:
                    break;
                }
            }
            if (iequals(call.library, "modload") && call.ordinal == 13u)
            {
                checkKelfPathCallback = {a0, cpu.gpr[28]};
                setV0(0);
                return ImportDisposition::Handled;
            }
            if (iequals(call.library, "ioman") && ioman.dispatchImport(call.ordinal, cpu, *this))
                return ImportDisposition::Handled;
            if (iequals(call.library, "sifman"))
            {
                const uint32_t callback = cpu.gpr[6];
                const uint32_t argument = cpu.gpr[7];
                if (!rpc.dispatchSifManImport(call.ordinal, cpu))
                    return ImportDisposition::Missing;
                // sceSifSetDmaIntr: the transfer completes at once; its callback runs like the
                // DMA completion interrupt (Dbc_Manager releases its DMA semaphore there).
                if (call.ordinal == 32u && cpu.gpr[2] != 0u && callback != 0u)
                    pendingGuestCallbacks.emplace(totalCycles + kSifDmaCompletionCycles,
                                                  ScheduledGuestCallback{callback, cpu.gpr[28], argument});
                return ImportDisposition::Handled;
            }
            if (iequals(call.library, "vblank") && vblank.dispatchImport(call.ordinal, cpu, totalCycles))
                return ImportDisposition::Handled;
            if (iequals(call.library, "timrman") && timrman.dispatchImport(call.ordinal, cpu, totalCycles))
                return ImportDisposition::Handled;
            if (iequals(call.library, "dmacman"))
            {
                dispatchDmacImport(call.ordinal, cpu);
                return ImportDisposition::Handled;
            }
            if (iequals(call.library, "stdio") && stdio.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;
            if (iequals(call.library, "sysclib"))
            {
                return sysclib.dispatchImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "heaplib") && heaplib.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;

            const uint32_t target = imports.resolve(call.library, call.ordinal, call.version);
            if (target != 0u)
            {
                cpu.pc = target;
                cpu.branchPending = false;
                return ImportDisposition::JumpToGuest;
            }

            std::ostringstream out;
            out << "[IOP] unhandled import " << call.library << ':' << call.ordinal
                << " version=0x" << std::hex << call.version << " pc=0x" << cpu.pc;
            log(LogLevel::Warning, out.str());
            setV0(0);
            return ImportDisposition::Missing;
        }

        bool step(CpuState &cpu)
        {
            if (cpu.stopped)
                return false;
            if (cpu.pc == kThreadReturnSentinel || cpu.pc == kCallReturnSentinel)
            {
                cpu.stopped = true;
                return false;
            }
            if (physicalAddress(cpu.pc) >= kRamSize)
            {
                std::ostringstream out;
                out << "[IOP] execution outside RAM pc=0x" << std::hex << cpu.pc;
                log(LogLevel::Error, out.str());
                cpu.stopped = true;
                return false;
            }
            if (checkInterrupt(cpu))
                return true;

            if (PcCounter *counter = pcCounter(); counter && cpu.pc >= counter->low && cpu.pc < counter->high)
            {
                ++counter->counts[(cpu.pc - counter->low) / 4u];
                if ((++counter->total & 0xFFFFFFu) == 0u)
                    counter->write();
            }
            if (const auto *traced = tracedPcs(); traced && traced->count(cpu.pc) != 0u)
                std::fprintf(stderr, "[iop-pc] pc=0x%x v1=0x%x v0=0x%x a0=0x%x a1=0x%x a2=0x%x a3=0x%x ra=0x%x\n", cpu.pc, cpu.gpr[3], cpu.gpr[2], cpu.gpr[4],
                             cpu.gpr[5], cpu.gpr[6], cpu.gpr[7], cpu.gpr[31]);

            if (const auto import = imports.decode(cpu.pc))
            {
                const ImportDisposition disposition = dispatchImport(*import, cpu);
                ++totalInstructions;
                ++totalCycles;
                if (disposition == ImportDisposition::JumpToGuest)
                    return true;
                cpu.pc = cpu.gpr[31];
                cpu.branchPending = false;
                return !cpu.stopped;
            }

            const bool running = cpuCore.executeInstruction(cpu);
            schedulePendingDma();
            ++totalInstructions;
            ++totalCycles;
            return running;
        }

        uint32_t runCpu(CpuState &cpu, uint32_t instructionBudget)
        {
            CpuState *previous = activeCpu;
            activeCpu = &cpu;
            const uint64_t start = totalInstructions;
            while (!cpu.stopped && !cpu.yielded && totalInstructions - start < instructionBudget)
            {
                if (!step(cpu))
                    break;
                if (totalCycles >= spu2.nextEventCycle())
                    spu2.advanceTo(totalCycles);
                if (!servicingDmaInterrupts && !pendingDmaInterrupts.empty())
                    servicePendingDmaInterrupts();
                if (!servicingGuestCallbacks && !pendingGuestCallbacks.empty())
                    servicePendingGuestCallbacks();
            }
            activeCpu = previous;
            return static_cast<uint32_t>(totalInstructions - start);
        }

        uint32_t callFunction(uint32_t address,
                              uint32_t a0,
                              uint32_t a1,
                              uint32_t a2,
                              uint32_t a3,
                              uint32_t gp,
                              uint32_t budget = kMaxCallInstructions)
        {
            struct CallDepthGuard
            {
                uint32_t &depth;
                ~CallDepthGuard() { --depth; }
            };

            const uint32_t depth = callDepth++;
            const CallDepthGuard depthGuard{callDepth};
            CpuState cpu{};
            cpu.pc = address;
            cpu.gpr[4] = a0;
            cpu.gpr[5] = a1;
            cpu.gpr[6] = a2;
            cpu.gpr[7] = a3;
            cpu.gpr[28] = gp;
            if (depth < kCallStackCapacity)
            {
                const uint32_t stackTop = kCallStackLimit - depth * kCallStackSize;
                cpu.gpr[29] = stackTop - 32u;
            }
            else if (activeCpu && activeCpu->gpr[29] > kCallStackBase + kStackGuardBytes)
            {
                // Extremely deep re-entrancy borrows unused space below the
                // suspended caller's live frame. Stack growth remains away
                // from the caller, so its saved registers stay intact.
                cpu.gpr[29] = (activeCpu->gpr[29] - kStackGuardBytes) & ~15u;
            }
            else
            {
                cpu.gpr[29] = kCallStackBase - 32u;
            }
            cpu.gpr[31] = kCallReturnSentinel;
            runCpu(cpu, budget);
            return cpu.gpr[2];
        }

        uint32_t executeGuestFunction(uint32_t address,
                                      uint32_t a0,
                                      uint32_t a1,
                                      uint32_t a2,
                                      uint32_t a3,
                                      uint32_t gp) override
        {
            return callFunction(address, a0, a1, a2, a3, gp);
        }

        uint32_t executeGuestFunctionWithBudget(uint32_t address,
                                                uint32_t a0,
                                                uint32_t a1,
                                                uint32_t a2,
                                                uint32_t a3,
                                                uint32_t gp,
                                                uint32_t instructionBudget) override
        {
            return callFunction(address, a0, a1, a2, a3, gp, instructionBudget);
        }

        // Not that good to use exception handling for control flow but will do for now
        void servicePendingDmaInterrupts()
        {
            if (servicingDmaInterrupts || pendingDmaInterrupts.empty())
                return;

            servicingDmaInterrupts = true;

            std::vector<int> completed;
            for (auto it = pendingDmaInterrupts.begin(); it != pendingDmaInterrupts.end();)
            {
                if (it->second > totalCycles)
                {
                    ++it;
                    continue;
                }
                completed.push_back(it->first);
                it = pendingDmaInterrupts.erase(it);
            }
            try
            {
                for (const int irq : completed)
                    (void)intrman.dispatchInterrupt(irq, *this);
            }
            catch (...)
            {
                servicingDmaInterrupts = false;
                throw;
            }
            servicingDmaInterrupts = false;
        }

        void servicePendingGuestCallbacks()
        {
            if (servicingGuestCallbacks || pendingGuestCallbacks.empty())
                return;

            std::vector<ScheduledGuestCallback> callbacks;
            for (auto it = pendingGuestCallbacks.begin(); it != pendingGuestCallbacks.end();)
            {
                if (it->first > totalCycles)
                    break;
                callbacks.push_back(it->second);
                it = pendingGuestCallbacks.erase(it);
            }
            if (callbacks.empty())
                return;

            servicingGuestCallbacks = true;
            try
            {
                for (const ScheduledGuestCallback &callback : callbacks)
                {
                    if (callback.function != 0u)
                    {
                        (void)callFunction(callback.function,
                                           callback.argument,
                                           0u,
                                           0u,
                                           0u,
                                           callback.gp,
                                           100000u);
                    }
                }
            }
            catch (...)
            {
                servicingGuestCallbacks = false;
                throw;
            }
            servicingGuestCallbacks = false;
        }

        // dmacman: only the SIO2 channels move data (dmac_request stores the transfer,
        // dmac_transfer runs it); the other calls succeed without effect.
        void dispatchDmacImport(uint16_t ordinal, CpuState &cpu)
        {
            const uint32_t channel = cpu.gpr[4];
            switch (ordinal)
            {
            case 28: // dmac_request(channel, address, words per block, blocks, direction)
                if (channel < dmaRequests.size())
                {
                    const uint32_t blocks = std::max<uint32_t>(cpu.gpr[7], 1u);
                    dmaRequests[channel] = {cpu.gpr[5], cpu.gpr[6] * blocks * 4u};
                }
                cpu.gpr[2] = 1u;
                return;
            case 32: // dmac_transfer(channel)
                if (channel == IopSio2::DmaInChannel)
                    sio2.dmaToDevice(dmaRequests[channel].address, dmaRequests[channel].bytes);
                else if (channel == IopSio2::DmaOutChannel)
                    sio2.armDmaFromDevice(dmaRequests[channel].address, dmaRequests[channel].bytes);
                cpu.gpr[2] = 0u;
                return;
            default:
                cpu.gpr[2] = 0u;
                return;
            }
        }

        bool queueSifCommand(uint32_t cid, const void *packet, uint32_t size)
        {
            if (!packet || size < 16u || size > kMaxSifCommandPacket)
                return false;
            const auto handler = rpc.cmdHandler(cid);
            if (!handler)
                return false;
            PendingSifCommand command{*handler, {}, size};
            std::memcpy(command.packet.data(), packet, size);
            pendingSifCommands.push_back(command);
            return true;
        }

        // Runs the handlers like the SIF command interrupt does: in order, outside any thread,
        // with the packet copied to an IOP receive buffer.
        void servicePendingSifCommands()
        {
            if (servicingSifCommands || pendingSifCommands.empty())
                return;
            if (sifCommandBuffer == 0u)
                sifCommandBuffer = memory.allocate(kMaxSifCommandPacket, 16u);
            if (sifCommandBuffer == 0u)
            {
                log(LogLevel::Error, "[IOP] no memory for the SIF command receive buffer");
                pendingSifCommands.clear();
                return;
            }
            servicingSifCommands = true;
            try
            {
                while (!pendingSifCommands.empty())
                {
                    const PendingSifCommand command = pendingSifCommands.front();
                    pendingSifCommands.pop_front();
                    if (memory.writeRam(sifCommandBuffer, command.packet.data(), command.size))
                    {
                        (void)callFunction(command.handler.function, sifCommandBuffer, command.handler.data, 0u, 0u,
                                           command.handler.gp, kSifCommandHandlerBudget);
                    }
                }
            }
            catch (...)
            {
                servicingSifCommands = false;
                throw;
            }
            servicingSifCommands = false;
        }

        void runCycles(uint64_t cycles) noexcept
        {
            try
            {
                const uint64_t target = totalCycles + cycles;
                while (totalCycles < target)
                {
                    spu2.advanceTo(totalCycles);
                    servicePendingDmaInterrupts();
                    servicePendingSifCommands();
                    servicePendingGuestCallbacks();
                    timrman.serviceDue(totalCycles, *this);
                    vblank.serviceDue(totalCycles, *this);
                    uint64_t nextDelayWake = UINT64_MAX;
                    IopThread *next = kernel.beginNextReady(totalCycles, &nextDelayWake);
                    if (!next)
                    {
                        uint64_t nextWake = std::min(target, nextDelayWake);
                        for (const auto &[irq, completionCycle] : pendingDmaInterrupts)
                            nextWake = std::min(nextWake, completionCycle);
                        if (!pendingGuestCallbacks.empty())
                            nextWake = std::min(nextWake, pendingGuestCallbacks.begin()->first);
                        nextWake = timrman.nextEventCycle(nextWake);
                        nextWake = vblank.nextEventCycle(nextWake);
                        nextWake = std::min(nextWake, spu2.nextEventCycle());
                        totalCycles = std::max(totalCycles + 1u, std::min(target, nextWake));
                        continue;
                    }
                    const uint64_t before = totalCycles;
                    runCpu(next->cpu, static_cast<uint32_t>(std::min<uint64_t>(kDefaultSlice, target - totalCycles)));
                    kernel.endTimeslice(*next, kThreadReturnSentinel);
                    if (totalCycles == before)
                        ++totalCycles;
                }
            }
            catch (...)
            {
                // Runtime scheduling must never throw through EeScheduler::accountCycles().
            }
        }

        ModuleLoadResult loadImage(std::string path, std::span<const uint8_t> image, const void *arguments, uint32_t argumentSize)
        {
            ModuleLoadResult result{true, -1, -1};
            const IopImageLoadResult loaded = IopModuleLoader::load(image, memory, moduleCursor);
            moduleCursor = loaded.nextModuleCursor;
            if (!loaded)
            {
                if (loaded.error == IopImageLoadError::InvalidElf)
                    log(LogLevel::Error, "[IOP] rejected invalid/non-MIPS IRX ELF");
                else if (loaded.error == IopImageLoadError::ArenaExhausted)
                    log(LogLevel::Error, "[IOP] module arena exhausted");
                return result;
            }
            if (!loaded.relocationsComplete)
                log(LogLevel::Warning, "[IOP] one or more IRX relocations were unsupported");

            Module module;
            module.id = nextModuleId++;
            module.path = std::move(path);
            const size_t slash = module.path.find_last_of("/\\:");
            module.name = slash == std::string::npos ? module.path : module.path.substr(slash + 1u);
            module.base = loaded.base;
            module.size = loaded.size;
            module.entry = loaded.entry;
            module.gp = loaded.gp;

            uint32_t args = 0u;
            if (arguments && argumentSize)
            {
                args = allocate(argumentSize + 1u, 16u);
                if (args)
                {
                    writeRam(args, arguments, argumentSize);
                    write8(args + argumentSize, 0u);
                }
            }
            const uint32_t startResult = callFunction(module.entry, argumentSize, args, 0u, 0u, module.gp);
            if (args)
                freeAllocation(args);
            module.resident = startResult == 0u || startResult == 2u;
            result.moduleId = module.id;
            result.startResult = static_cast<int32_t>(startResult);
            modules[module.id] = std::move(module);

            std::ostringstream out;
            out << "[IOP] loaded IRX id=" << result.moduleId
                << " entry=0x" << std::hex << modules[result.moduleId].entry
                << " base=0x" << modules[result.moduleId].base
                << " start=" << std::dec << result.startResult;
            log(LogLevel::Info, out.str());
            return result;
        }

        ModuleLoadResult loadModule(std::string_view path, const void *arguments, uint32_t argumentSize)
        {
            std::vector<uint8_t> image;
            if (!IopModuleLoader::readWholeHostFile(host, path, image))
            {
                log(LogLevel::Warning, std::string("[IOP] failed to open IRX '") + std::string(path) + "'");
                return {true, -1, -1};
            }
            return loadImage(std::string(path), image, arguments, argumentSize);
        }

        ModuleLoadResult loadModuleBuffer(uint32_t guestAddress, const void *arguments, uint32_t argumentSize)
        {
            std::vector<uint8_t> image;
            if (!IopModuleLoader::readElfFromGuest(host, guestAddress, image))
                return {true, -1, -1};
            std::ostringstream tag;
            tag << "buffer@0x" << std::hex << guestAddress;
            return loadImage(tag.str(), image, arguments, argumentSize);
        }

        ModuleLoadResult loadModuleBufferFromIop(uint32_t iopAddress, const void *arguments, uint32_t argumentSize)
        {
            std::vector<uint8_t> image;
            if (!IopModuleLoader::readElfFromIop(memory, iopAddress, image))
                return {true, -1, -1};
            std::ostringstream tag;
            tag << "iop-buffer@0x" << std::hex << iopAddress;
            return loadImage(tag.str(), image, arguments, argumentSize);
        }

        bool stopModule(int32_t moduleId, int32_t *result)
        {
            auto it = modules.find(moduleId);
            if (it == modules.end())
                return false;
            // A removable IRX normally exposes a stop entry through module metadata. We do not guess it; terminate owned execution and release the image cleanly.
            kernel.terminateThreadsInRange(it->second.base, it->second.size);
            rpc.removeServersInRange(it->second.base, it->second.size);
            imports.eraseRange(it->second.base, it->second.size);
            modules.erase(it);
            kernel.cleanupDeadThreads();
            if (result)
                *result = 0;
            return true;
        }

        IopHost &host;
        IopMemory memory;
        IopSysmem sysmem;
        IopKernel kernel;
        IopCdvd cdvd;
        IopVblank vblank;
        IopRpcBridge rpc;
        IopSysclib sysclib;
        IopStdio stdio;
        IopHeaplib heaplib;
        IopIntrman intrman;
        IopTimrman timrman;
        IopIoman ioman;
        IopCpuCore cpuCore;
        IopImportRegistry imports;
        IopLoadcore loadcore;
        std::map<int, Module> modules;
        std::map<int, uint64_t> pendingDmaInterrupts;
        std::multimap<uint64_t, ScheduledGuestCallback> pendingGuestCallbacks;
        IopSio2 sio2;
        IopSpu2 spu2;
        struct DmaRequest
        {
            uint32_t address = 0u;
            uint32_t bytes = 0u;
        };
        std::array<DmaRequest, 16> dmaRequests{};
        std::deque<PendingSifCommand> pendingSifCommands;
        uint32_t sifCommandBuffer = 0u;
        bool servicingSifCommands = false;
        uint32_t nextModuleId = 1;
        uint32_t moduleCursor = kModuleLoadBase;
        uint64_t totalCycles = 0;
        uint64_t totalInstructions = 0;
        uint64_t eeCycleCarry = 0;
        CpuState *activeCpu = nullptr;
        std::string lastError;
        bool servicingDmaInterrupts = false;
        bool servicingGuestCallbacks = false;
        uint32_t callDepth = 0u;
        GuestCallback secrMcCommandHandler;
        GuestCallback secrMcDevIdHandler;
        GuestCallback checkKelfPathCallback;
    };

    IopEmulator::IopEmulator(IopHost &host)
        : m_impl(std::make_unique<Impl>(host))
    {
    }

    IopEmulator::~IopEmulator() = default;

    void IopEmulator::reset()
    {
        m_impl->reset();
    }

    ModuleLoadResult IopEmulator::loadModule(std::string_view path, const void *arguments, uint32_t argumentSize)
    {
        return m_impl->loadModule(path, arguments, argumentSize);
    }

    ModuleLoadResult IopEmulator::loadModuleBuffer(uint32_t guestAddress, const void *arguments, uint32_t argumentSize)
    {
        return m_impl->loadModuleBuffer(guestAddress, arguments, argumentSize);
    }

    ModuleLoadResult IopEmulator::loadModuleBufferFromIop(uint32_t iopAddress, const void *arguments, uint32_t argumentSize)
    {
        return m_impl->loadModuleBufferFromIop(iopAddress, arguments, argumentSize);
    }

    bool IopEmulator::sendSifCommand(uint32_t cid, const void *packet, uint32_t size)
    {
        return m_impl->queueSifCommand(cid, packet, size);
    }

    bool IopEmulator::stopModule(int32_t moduleId, int32_t *result)
    {
        return m_impl->stopModule(moduleId, result);
    }

    void IopEmulator::runEeCycles(uint64_t eeCycles) noexcept
    {
        // The EE accounts a few dozen cycles per checkpoint: running the IOP scheduler that
        // often cost a third of the EE thread. Batch 256 IOP cycles (about 7 us) per run.
        constexpr uint64_t kRunBatchEeCycles = 8u * 256u;
        const uint64_t total = m_impl->eeCycleCarry + eeCycles;
        if (total < kRunBatchEeCycles)
        {
            m_impl->eeCycleCarry = total;
            return;
        }
        m_impl->eeCycleCarry = total % 8u;
        m_impl->runCycles(total / 8u);
    }

    RpcResult IopEmulator::handleRpc(const RpcRequest &request)
    {
        return m_impl->rpc.handleRpc(request, *m_impl);
    }

    bool IopEmulator::hasRpcServer(uint32_t sid) const noexcept
    {
        return m_impl->rpc.hasServer(sid);
    }

    void IopEmulator::onSifTransfer(const SifTransfer &transfer)
    {
        m_impl->rpc.onSifTransfer(transfer);
    }

    uint32_t IopEmulator::allocateMemory(uint32_t size, uint32_t alignment)
    {
        return m_impl->memory.allocate(size, alignment);
    }

    bool IopEmulator::freeMemory(uint32_t address)
    {
        return m_impl->memory.freeAllocation(address);
    }

    bool IopEmulator::readMemory(uint32_t address, void *destination, size_t size) const
    {
        return isMemoryRange(address, size) &&
               m_impl->memory.readRam(address, destination, size);
    }

    bool IopEmulator::writeMemory(uint32_t address, const void *source, size_t size)
    {
        return isMemoryRange(address, size) &&
               m_impl->memory.writeRam(address, source, size);
    }

    bool IopEmulator::zeroMemory(uint32_t address, size_t size)
    {
        return isMemoryRange(address, size) &&
               m_impl->memory.zeroRam(address, size);
    }

    bool IopEmulator::isMemoryRange(uint32_t address, size_t size) const
    {
        const bool physicalSegment = address < IopMemory::RamSize;
        const bool cachedSegment = address >= 0x80000000u && address < 0x80200000u;
        const bool uncachedSegment = address >= 0xA0000000u && address < 0xA0200000u;
        if (!physicalSegment && !cachedSegment && !uncachedSegment)
            return false;
        const uint32_t physical = IopMemory::physicalAddress(address);
        return physical <= IopMemory::RamSize && size <= IopMemory::RamSize - physical;
    }

    uint64_t IopEmulator::cycles() const noexcept
    {
        return m_impl->totalCycles;
    }

    uint64_t IopEmulator::instructions() const noexcept
    {
        return m_impl->totalInstructions;
    }

    uint32_t IopEmulator::loadedModuleCount() const noexcept
    {
        return static_cast<uint32_t>(m_impl->modules.size());
    }

    std::vector<std::string> IopEmulator::describeThreads() const
    {
        return m_impl->kernel.describeThreads();
    }

    std::vector<std::string> IopEmulator::describeSpu2() const
    {
        return m_impl->spu2.describe();
    }

    uint32_t IopEmulator::threadCount() const noexcept
    {
        return static_cast<uint32_t>(m_impl->kernel.threadCount());
    }

    uint32_t IopEmulator::rpcServerCount() const noexcept
    {
        return static_cast<uint32_t>(m_impl->rpc.serverCount());
    }

}
