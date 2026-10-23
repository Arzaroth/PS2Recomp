#ifndef PS2_VU1_STEP_H
#define PS2_VU1_STEP_H

// The interpreter's per-pair step, inline so code generated for a known microprogram can constant-fold it
// for each decoded pair.
#include "runtime/ps2_vu1.h"
#include "ps2_vu1_lower.h"
#include "ps2_vu1_upper.h"

#include <algorithm>
#include <cstring>
#include <immintrin.h>

__attribute__((always_inline)) inline float VU1Interpreter::normalizeOperand(float value) const
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t exponent = (bits >> 23) & 0xFFu;
    if (exponent == 0u)
    {
        bits &= 0x80000000u;
    }
    else if (exponent == 0xFFu)
    {
        bits = (bits & 0x80000000u) | 0x7F7FFFFFu;
    }
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

__attribute__((always_inline)) inline float VU1Interpreter::normalizeResult(float value, uint32_t &laneFlags) const
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t magnitude = bits & 0x7FFFFFFFu;
    const uint32_t exponent = (bits >> 23) & 0xFFu;

    laneFlags = sign != 0u ? 0x2u : 0u;
    if (magnitude == 0u)
    {
        laneFlags |= 0x1u;
    }
    else if (exponent == 0u)
    {
        laneFlags |= 0x5u;
        bits = sign;
    }
    else if (exponent == 0xFFu)
    {
        laneFlags |= 0x8u;
        bits = sign | 0x7F7FFFFFu;
    }

    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

__attribute__((always_inline)) inline void VU1Interpreter::applyDest(float *dst, const float *result, uint8_t dest)
{
    if (dest & 0x8u)
        dst[0] = result[0];
    if (dest & 0x4u)
        dst[1] = result[1];
    if (dest & 0x2u)
        dst[2] = result[2];
    if (dest & 0x1u)
        dst[3] = result[3];
}

__attribute__((always_inline)) inline void VU1Interpreter::applyDestAcc(const float *result, uint8_t dest)
{
    applyDest(m_state.acc, result, dest);
}

inline uint32_t VU1Interpreter::microAddressMask() const
{
    return m_unit == Unit::VU1 ? 0x3FFFu : 0x0FFFu;
}

inline int32_t VU1Interpreter::readBranchVi(uint8_t reg) const
{
    if (reg == 0u)
        return 0;
    if (m_viBranchBackupValid &&
        m_viBranchBackupReg == reg)
    {
        return m_viBranchBackupValue;
    }
    return m_state.vi[reg];
}

inline void VU1Interpreter::recordViWriteForBranch(uint8_t reg, int32_t oldValue)
{
    if (reg == 0u)
        return;
    m_viBranchBackupValue = oldValue;
    m_viBranchBackupReg = reg;
    m_viBranchBackupValid = true;
}


__attribute__((always_inline)) inline void VU1Interpreter::advanceOneCycle()
{
    advanceTo(m_cycle + 1u);
}

__attribute__((always_inline)) inline void VU1Interpreter::advanceTo(uint64_t targetCycle)
{
    if (targetCycle <= m_cycle)
        return;
    // Nothing executes while the clock jumps, so pending results and PATH1 progress can be applied in one step.
    const uint64_t elapsed = targetCycle - m_cycle;
    m_cycle = targetCycle;
    m_state.cycles = m_cycle;
    if (m_cycle >= m_nextFlagCommit || m_cycle >= m_nextScalarCommit)
        commitReadyPipelines();
    if (m_xgkick.active)
        m_xgkickBacklog += elapsed;
}

__attribute__((always_inline)) inline uint64_t VU1Interpreter::calculatePairReadyCycle(const DecodedInstructionPair &decoded) const
{
    uint64_t ready = m_cycle;
    const auto laneReady = [&ready](const std::array<uint64_t, 4> &laneCycles, uint8_t lanes) __attribute__((always_inline))
    {
        if (lanes & 0x8u)
            ready = std::max(ready, laneCycles[0]);
        if (lanes & 0x4u)
            ready = std::max(ready, laneCycles[1]);
        if (lanes & 0x2u)
            ready = std::max(ready, laneCycles[2]);
        if (lanes & 0x1u)
            ready = std::max(ready, laneCycles[3]);
    };
    const auto vfReady = [&](const InstructionUsage &usage) __attribute__((always_inline))
    {
        for (uint32_t index = 0; index < usage.vfReadCount; ++index)
        {
            const uint8_t reg = usage.vfRead[index].reg;
            if (m_vfReadyMax[reg] > ready)
                laneReady(m_vfReady[reg], usage.vfRead[index].lanes);
        }
    };
    vfReady(decoded.upperUsage);
    vfReady(decoded.lowerUsage);
    for (uint32_t mask = (decoded.upperUsage.viRead | decoded.lowerUsage.viRead) & 0xFFFEu; mask != 0u; mask &= mask - 1u)
        ready = std::max(ready, m_viReady[static_cast<uint32_t>(_tzcnt_u32(mask))]);
    if (const uint8_t accRead = decoded.upperUsage.accRead | decoded.lowerUsage.accRead; accRead != 0u)
        laneReady(m_accReady, accRead);

    if (decoded.lowerUsage.pipeline == PipelineFdiv && m_fdiv.valid)
        ready = std::max(ready, m_fdiv.readyCycle);
    if (decoded.lowerUsage.pipeline == PipelineEfu)
        ready = std::max(ready, m_efuResourceReady);
    if (decoded.lowerUsage.waitQ && m_fdiv.valid)
        ready = std::max(ready, m_fdiv.readyCycle);
    if (decoded.lowerUsage.waitP)
    {
        for (const ScalarPipelineEntry &entry : m_efu)
            if (entry.valid)
                ready = std::max(ready, entry.readyCycle);
    }
    if (decoded.lowerUsage.pipeline == PipelineXgkick && m_xgkick.active)
        ready = std::max(ready, m_cycle + 1u);
    return ready;
}

__attribute__((always_inline)) inline void VU1Interpreter::markPairWrites(const DecodedInstructionPair &decoded)
{
    // Writeback completion of everything issued so far; the end-of-program flush waits for it.
    const auto noteWriteback = [this](uint64_t readyCycle)
    {
        if (readyCycle > m_lastWritebackCycle)
            m_lastWritebackCycle = readyCycle;
    };
    const auto setLanes = [](std::array<uint64_t, 4> &laneCycles, uint8_t lanes, uint64_t readyCycle)
    {
        if (lanes & 0x8u)
            laneCycles[0] = readyCycle;
        if (lanes & 0x4u)
            laneCycles[1] = readyCycle;
        if (lanes & 0x2u)
            laneCycles[2] = readyCycle;
        if (lanes & 0x1u)
            laneCycles[3] = readyCycle;
    };

    const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
    if (lowerWrite.reg != 0u &&
        decoded.suppressedLowerVf != lowerWrite.reg)
    {
        const uint32_t latency = decoded.lowerUsage.vfLatency != 0u
                                     ? decoded.lowerUsage.vfLatency
                                     : decoded.lowerUsage.latency;
        setLanes(m_vfReady[lowerWrite.reg], lowerWrite.lanes, m_cycle + latency);
        m_vfReadyMax[lowerWrite.reg] = std::max(m_vfReadyMax[lowerWrite.reg], m_cycle + latency);
        noteWriteback(m_cycle + latency);
    }

    const VfAccess upperWrite = decoded.upperUsage.vfWrite;
    if (upperWrite.reg != 0u)
    {
        const uint32_t latency = decoded.upperUsage.vfLatency != 0u
                                     ? decoded.upperUsage.vfLatency
                                     : decoded.upperUsage.latency;
        setLanes(m_vfReady[upperWrite.reg], upperWrite.lanes, m_cycle + latency);
        m_vfReadyMax[upperWrite.reg] = std::max(m_vfReadyMax[upperWrite.reg], m_cycle + latency);
        noteWriteback(m_cycle + latency);
    }

    if (const uint32_t viWrites = decoded.lowerUsage.viWrite & 0xFFFEu; viWrites != 0u)
    {
        const uint64_t readyCycle = m_cycle + (decoded.lowerUsage.viLatency != 0u ? decoded.lowerUsage.viLatency
                                                                                   : decoded.lowerUsage.latency);
        for (uint32_t mask = viWrites; mask != 0u; mask &= mask - 1u)
            m_viReady[static_cast<uint32_t>(_tzcnt_u32(mask))] = readyCycle;
        noteWriteback(readyCycle);
    }
    if (decoded.upperUsage.accWrite != 0u)
    {
        setLanes(m_accReady, decoded.upperUsage.accWrite, m_cycle + kAccForwardLatency);
        noteWriteback(m_cycle + kAccForwardLatency);
    }
}

template <bool Sequential, typename Upper, typename Lower>
__attribute__((always_inline)) inline VU1Interpreter::StepResult VU1Interpreter::stepPair(const DecodedInstructionPair &decoded,
                                                                                    uint64_t budgetEnd, uint32_t codeSize,
                                                                                    Upper &&upper, Lower &&lower)
{
    if (decoded.upperUsage.reserved || decoded.lowerUsage.reserved)
    {
        reportReservedInstruction(decoded.upperUsage.reserved, decoded.upperUsage.reserved ? decoded.upper : decoded.lower);
        return StepResult::Stop;
    }

    const bool kicks = decoded.lowerUsage.pipeline == PipelineXgkick;
    if (kicks)
        catchUpXgkick();
    uint64_t readyCycle = calculatePairReadyCycle(decoded);
    while (readyCycle > m_cycle)
    {
        if (readyCycle >= budgetEnd)
        {
            advanceTo(budgetEnd);
            break;
        }
        advanceTo(readyCycle);
        if (kicks)
            catchUpXgkick();
        readyCycle = calculatePairReadyCycle(decoded);
    }
    if (m_cycle >= budgetEnd)
        return StepResult::Stop;

    const uint8_t writtenVi = decoded.writtenVi;
    const int32_t oldVi = writtenVi != 0u ? m_state.vi[writtenVi] : 0;

    // Results are written immediately. Every reader stalls until its operands' latency has elapsed
    // (calculatePairReadyCycle), so the delayed writeback is unobservable inside a run; only the same-pair
    // case (lower reads the register upper writes) needs the old value. A run that stops on its cycle
    // budget hides writes still in flight (logInFlightWrites / hideInFlightWrites).
    const VfAccess upperWrite = decoded.upperUsage.vfWrite;
    const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
    const bool logLowerVf = lowerWrite.reg != 0u && decoded.suppressedLowerVf != lowerWrite.reg &&
                            lowerWrite.reg != upperWrite.reg;
    alignas(16) float oldUpperVf[4];
    alignas(16) float oldLowerVf[4];
    alignas(16) float oldAcc[4];
    if (m_logInFlight)
    {
        if (upperWrite.reg != 0u)
            std::memcpy(oldUpperVf, m_state.vf[upperWrite.reg], sizeof(oldUpperVf));
        if (logLowerVf)
            std::memcpy(oldLowerVf, m_state.vf[lowerWrite.reg], sizeof(oldLowerVf));
        if (decoded.upperUsage.accWrite != 0u)
            std::memcpy(oldAcc, m_state.acc, sizeof(oldAcc));
    }

    if (decoded.iBit)
    {
        upper();
        float immediate = 0.0f;
        std::memcpy(&immediate, &decoded.lower, sizeof(immediate));
        m_state.i = normalizeOperand(immediate);
    }
    else if (decoded.upperVfShadowReg != 0u)
    {
        float oldVf[4]{};
        float upperVf[4]{};
        std::memcpy(oldVf,
                    m_state.vf[decoded.upperVfShadowReg],
                    sizeof(oldVf));
        upper();
        std::memcpy(upperVf,
                    m_state.vf[decoded.upperVfShadowReg],
                    sizeof(upperVf));
        std::memcpy(m_state.vf[decoded.upperVfShadowReg],
                    oldVf,
                    sizeof(oldVf));
        lower();
        std::memcpy(m_state.vf[decoded.upperVfShadowReg],
                    upperVf,
                    sizeof(upperVf));
    }
    else
    {
        upper();
        lower();
    }

    m_viBranchBackupValid = false;

    markPairWrites(decoded);
    if (m_logInFlight && upperWrite.reg != 0u)
    {
        const uint32_t latency =
            decoded.upperUsage.vfLatency != 0u ? decoded.upperUsage.vfLatency : decoded.upperUsage.latency;
        logInFlightWrite(InFlightWrite::Kind::Vf, upperWrite.reg, upperWrite.lanes, m_cycle + latency, oldUpperVf,
                         m_state.vf[upperWrite.reg]);
    }
    if (m_logInFlight && logLowerVf)
    {
        const uint32_t latency =
            decoded.lowerUsage.vfLatency != 0u ? decoded.lowerUsage.vfLatency : decoded.lowerUsage.latency;
        logInFlightWrite(InFlightWrite::Kind::Vf, lowerWrite.reg, lowerWrite.lanes, m_cycle + latency, oldLowerVf,
                         m_state.vf[lowerWrite.reg]);
    }
    if (m_logInFlight && decoded.upperUsage.accWrite != 0u)
        logInFlightWrite(InFlightWrite::Kind::Acc, 0u, decoded.upperUsage.accWrite, m_cycle + kAccForwardLatency,
                         oldAcc, m_state.acc);
    if (m_logInFlight && writtenVi != 0u)
    {
        const uint32_t latency =
            decoded.lowerUsage.viLatency != 0u ? decoded.lowerUsage.viLatency : decoded.lowerUsage.latency;
        const float oldValue[4] = {static_cast<float>(oldVi), 0.0f, 0.0f, 0.0f};
        const float newValue[4] = {static_cast<float>(m_state.vi[writtenVi]), 0.0f, 0.0f, 0.0f};
        logInFlightWrite(InFlightWrite::Kind::Vi, writtenVi, 0x8u, m_cycle + latency, oldValue, newValue);
    }
    if (writtenVi != 0u && decoded.lowerUsage.delaysNextBranchRead)
        recordViWriteForBranch(writtenVi, oldVi);

    m_state.vf[0][0] = 0.0f;
    m_state.vf[0][1] = 0.0f;
    m_state.vf[0][2] = 0.0f;
    m_state.vf[0][3] = 1.0f;
    m_state.vi[0] = 0;

    if constexpr (Sequential)
    {
        m_state.pc += 8u;
        advanceOneCycle();
        return StepResult::Next;
    }

    uint32_t nextPc = m_state.pc + 8u;
    if (nextPc >= codeSize)
        nextPc = 0u;
    m_state.pc = nextPc;

    if (m_state.branchPending)
    {
        if (m_state.branchDelay == 0u)
        {
            m_state.pc = m_state.branchTarget & microAddressMask();
            m_state.branchPending = false;
        }
        else
        {
            --m_state.branchDelay;
        }
    }

    bool programEnded = false;
    const bool dHalt = decoded.dBit && m_state.dBitEnabled;
    const bool tHalt = decoded.tBit && m_state.tBitEnabled;
    const bool haltBit = dHalt || tHalt;
    const bool haltBranch = haltBit && decoded.lowerUsage.pipeline == PipelineBranch;

    if (m_state.haltAfterDelaySlot)
    {
        m_state.stoppedByD = m_pendingHaltD;
        m_state.stoppedByT = m_pendingHaltT;
        programEnded = true;
    }
    else if (m_state.ebit)
        programEnded = true;
    else if (haltBit && !haltBranch)
    {
        m_state.stoppedByD = dHalt;
        m_state.stoppedByT = tHalt;
        programEnded = true;
    }
    else if (decoded.eBit)
        m_state.ebit = true;
    else if (haltBranch)
    {
        m_state.haltAfterDelaySlot = true;
        m_pendingHaltD = dHalt;
        m_pendingHaltT = tHalt;
    }

    advanceOneCycle();
    return programEnded ? StepResult::Ended : StepResult::Next;
}

// What code generated by ps2_vu1_compile uses of the interpreter.
struct Vu1CompiledAccess
{
    using Pair = VU1Interpreter::DecodedInstructionPair;
    using Usage = VU1Interpreter::InstructionUsage;
    using Access = VU1Interpreter::VfAccess;
    using Pipeline = VU1Interpreter::Pipeline;
    using Result = VU1Interpreter::StepResult;
    using Kind = VU1Interpreter::FmacKind;
    using Operand = VU1Interpreter::FmacOperand;
    using UpperHandler = VU1Interpreter::UpperHandler;
    using Run = VU1Interpreter::CompiledRun;

    static bool running(const VU1Interpreter &vu, uint64_t budgetEnd) { return vu.m_cycle < budgetEnd && !vu.m_stopRequested; }
    static uint32_t pc(const VU1Interpreter &vu) { return vu.m_state.pc; }

    template <bool Sequential, typename Upper, typename Lower>
    __attribute__((always_inline)) static Result step(VU1Interpreter &vu, const Pair &decoded, bool flagsLive, uint64_t budgetEnd,
                                                      uint32_t codeSize, Upper &&upper, Lower &&lower)
    {
        vu.m_currentFlagsLive = flagsLive;
        return vu.stepPair<Sequential>(decoded, budgetEnd, codeSize, upper, lower);
    }
    template <Kind K, Operand O, bool ToAcc>
    __attribute__((always_inline)) static void fmac(VU1Interpreter &vu, uint32_t upper)
    {
        [[clang::always_inline]] vu.execFmac<K, O, ToAcc>(upper);
    }
    __attribute__((always_inline)) static void upperNop(VU1Interpreter &vu, uint32_t upper) { vu.execUpperNop(upper); }
    __attribute__((always_inline)) static void upperGeneric(VU1Interpreter &vu, uint32_t upper)
    {
        [[clang::always_inline]] vu.execUpper(upper);
    }
    __attribute__((always_inline)) static void lower(VU1Interpreter &vu, uint32_t lower, uint8_t *vuData, uint32_t dataSize, GS &gs,
                                                     PS2Memory *memory, uint32_t upper)
    {
        [[clang::always_inline]] vu.execLower(lower, vuData, dataSize, gs, memory, upper);
    }

    static void registerProgram(uint64_t hash, uint32_t codeSize, Run run)
    {
        VU1Interpreter::compiledPrograms().push_back({hash, codeSize, run});
    }

    // For the generator.
    static uint64_t hash(const uint8_t *code, uint32_t size) { return VU1Interpreter::hashMicrocode(code, size); }
    static void flagLiveness(const uint8_t *code, uint32_t size, bool *live)
    {
        VU1Interpreter::computeMicrocodeFlagLiveness(code, size, size - 1u, live);
    }
    static Pair decode(const VU1Interpreter &vu, const uint8_t *code, uint32_t pc) { return vu.decodeInstructionPair(code, pc); }
    static UpperHandler upperHandler(uint32_t upper) { return VU1Interpreter::selectUpperHandler(upper); }
    static UpperHandler nopHandler() { return &VU1Interpreter::execUpperNop; }
    static UpperHandler genericHandler() { return &VU1Interpreter::execUpper; }
    template <Kind K, Operand O, bool ToAcc>
    static UpperHandler fmacHandler() { return &VU1Interpreter::execFmac<K, O, ToAcc>; }
};

#endif
