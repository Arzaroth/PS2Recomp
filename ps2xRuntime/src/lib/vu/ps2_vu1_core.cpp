#include "runtime/ps2_vu1.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/ps2_memory.h"
#include "ps2_vu1_detail.h"
#include "ps2_vu1_step.h"

#include <algorithm>
#include <immintrin.h>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <ps2_log.h>

namespace
{
    constexpr uint8_t laneForComponent(uint32_t component)
    {
        return static_cast<uint8_t>(1u << (3u - component));
    }

    // Upper instructions that queue MAC and status flags (applyFmacFlags).
    bool writesFmacFlags(uint32_t upper)
    {
        const uint32_t op = upper & 0x3Fu;
        const uint32_t code = op >= 0x3Cu ? ((upper & 3u) | ((upper >> 4) & 0x7Cu)) : op;
        if (op >= 0x30u && op < 0x3Cu)
            return false;
        return code <= 0x0Fu || (code >= 0x18u && code <= 0x1Cu) || code == 0x1Eu || (code >= 0x20u && code <= 0x2Au) ||
               (code >= 0x2Cu && code <= 0x2Eu);
    }

    uint64_t hashCode(const uint8_t *code, uint32_t size)
    {
        uint64_t hash = 0xCBF29CE484222325ull ^ size;
        for (uint32_t offset = 0; offset + 8u <= size; offset += 8u)
        {
            uint64_t word = 0;
            std::memcpy(&word, code + offset, sizeof(word));
            hash = (hash ^ word) * 0x100000001B3ull;
            hash ^= hash >> 29;
        }
        return hash;
    }

    // A flag reader issued at cycle t sees the last flag write issued at or before t - 4, so an FMAC's flags
    // are dead once four newer flag writers issued on every path before a reader. Program ends, halts and
    // jumps to registers count as readers.
    void computeFlagLiveness(const uint8_t *code, uint32_t codeSize, uint32_t pcMask, bool *live)
    {
        const uint32_t pairs = std::min<uint32_t>(codeSize / 8u, (pcMask + 1u) / 8u);
        if (pairs == 0u)
            return;
        constexpr uint32_t kEnd = UINT32_MAX;
        std::vector<uint8_t> writer(pairs), reader(pairs);
        std::vector<std::array<uint32_t, 2>> successors(pairs);
        bool statusRead = false, irregular = false;
        for (uint32_t index = 0; index < pairs; ++index)
        {
            uint32_t lower = 0, upper = 0;
            std::memcpy(&lower, code + index * 8u, sizeof(lower));
            std::memcpy(&upper, code + index * 8u + 4u, sizeof(upper));
            writer[index] = writesFmacFlags(upper);
            successors[index] = {(index + 1u) % pairs, kEnd - 1u};
            if ((upper & 0x18000000u) != 0u)
                successors[index][0] = kEnd;
            if ((upper & 0x80000000u) != 0u)
                continue;
            const uint32_t op = (lower >> 25) & 0x7Fu;
            reader[index] = op == 0x18u || op == 0x1Au || op == 0x1Bu;
            statusRead |= op == 0x14u || op == 0x16u || op == 0x17u;
        }
        for (uint32_t index = 0; index < pairs; ++index)
        {
            uint32_t lower = 0, upper = 0;
            std::memcpy(&lower, code + index * 8u, sizeof(lower));
            std::memcpy(&upper, code + index * 8u + 4u, sizeof(upper));
            const uint32_t slot = (index + 1u) % pairs;
            if ((upper & 0x40000000u) != 0u)
                successors[slot] = {kEnd, kEnd};
            if ((upper & 0x80000000u) != 0u)
                continue;
            const uint32_t op = (lower >> 25) & 0x7Fu;
            const bool branch = op == 0x20u || op == 0x21u || op == 0x28u || op == 0x29u || (op >= 0x2Cu && op <= 0x2Fu);
            const bool jump = op == 0x24u || op == 0x25u;
            if (!branch && !jump)
                continue;
            uint32_t slotLower = 0, slotUpper = 0;
            std::memcpy(&slotLower, code + slot * 8u, sizeof(slotLower));
            std::memcpy(&slotUpper, code + slot * 8u + 4u, sizeof(slotUpper));
            const uint32_t slotOp = (slotLower >> 25) & 0x7Fu;
            if ((slotUpper & 0x80000000u) == 0u && slotOp >= 0x20u && slotOp <= 0x2Fu)
                irregular = true;
            const int32_t offset = static_cast<int32_t>(static_cast<int16_t>(static_cast<uint16_t>(lower << 5)) >> 5);
            successors[slot][1] = jump ? kEnd : ((index * 8u + 8u + static_cast<uint32_t>(offset * 8)) & pcMask) / 8u;
            if (successors[slot][1] != kEnd && successors[slot][1] >= pairs)
                successors[slot][1] = kEnd;
        }
        if (statusRead || irregular)
        {
            std::fill(live, live + pairs, true);
            return;
        }

        std::vector<uint8_t> visited(pairs);
        std::vector<std::pair<uint32_t, uint8_t>> stack;
        for (uint32_t start = 0; start < pairs; ++start)
        {
            live[start] = false;
            if (!writer[start])
                continue;
            std::fill(visited.begin(), visited.end(), 0u);
            stack.clear();
            const auto push = [&](uint32_t from, uint8_t newer)
            {
                for (const uint32_t next : successors[from])
                {
                    if (next == kEnd - 1u)
                        continue;
                    if (next == kEnd)
                        return false;
                    if ((visited[next] & (1u << newer)) == 0u)
                    {
                        visited[next] |= static_cast<uint8_t>(1u << newer);
                        stack.emplace_back(next, newer);
                    }
                }
                return true;
            };
            bool observed = !push(start, 0u);
            while (!observed && !stack.empty())
            {
                const auto [index, newer] = stack.back();
                stack.pop_back();
                if (reader[index])
                {
                    observed = true;
                    break;
                }
                const uint8_t after = static_cast<uint8_t>(newer + (writer[index] ? 1u : 0u));
                if (after < 4u)
                    observed = !push(index, after);
            }
            live[start] = observed;
        }
    }
}

void VU1Interpreter::addVfRead(InstructionUsage &usage, uint8_t reg, uint8_t lanes)
{
    if (lanes == 0u)
        return;
    for (uint32_t index = 0; index < usage.vfReadCount; ++index)
    {
        if (usage.vfRead[index].reg == reg)
        {
            usage.vfRead[index].lanes |= lanes;
            return;
        }
    }
    if (usage.vfReadCount < usage.vfRead.size())
        usage.vfRead[usage.vfReadCount++] = {reg, lanes};
}

void VU1Interpreter::addVfWrite(InstructionUsage &usage, uint8_t reg, uint8_t lanes)
{
    if (reg == 0u || lanes == 0u)
        return;
    if (usage.vfWrite.reg == 0u)
        usage.vfWrite = {reg, lanes};
    else if (usage.vfWrite.reg == reg)
        usage.vfWrite.lanes |= lanes;
}

uint8_t VU1Interpreter::vfReadLanes(const InstructionUsage &usage, uint8_t reg)
{
    for (uint32_t index = 0; index < usage.vfReadCount; ++index)
    {
        if (usage.vfRead[index].reg == reg)
            return usage.vfRead[index].lanes;
    }
    return 0u;
}

VU1Interpreter::VU1Interpreter(Unit unit)
    : m_unit(unit)
{
    const char *exactFlags = std::getenv("PS2X_VU1_EXACT_FLAGS");
    m_elideDeadFlags = unit == Unit::VU1 && !(exactFlags && *exactFlags && *exactFlags != '0');
    reset();
}

std::vector<std::pair<uint64_t, VU1Interpreter::CompiledRun>> &VU1Interpreter::compiledPrograms()
{
    static std::vector<std::pair<uint64_t, CompiledRun>> programs;
    return programs;
}

uint64_t VU1Interpreter::hashMicrocode(const uint8_t *code, uint32_t size)
{
    return hashCode(code, size);
}

void VU1Interpreter::computeMicrocodeFlagLiveness(const uint8_t *code, uint32_t size, uint32_t pcMask, bool *live)
{
    std::fill(live, live + std::min<uint32_t>(size / 8u, kMaxDecodedPairs), true);
    computeFlagLiveness(code, size, pcMask, live);
}

void VU1Interpreter::refreshFlagLiveness(const uint8_t *vuCode, uint32_t codeSize, uint64_t stamp)
{
    m_flagsLiveStamp = stamp;
    const uint64_t hash = hashCode(vuCode, codeSize);
    for (const CodeImageInfo &image : m_flagLivenessCache)
        if (image.hash == hash)
        {
            m_flagsLive = &image.liveness;
            m_compiledRun = image.compiled;
            return;
        }
    if (m_flagLivenessCache.size() >= 16u)
        m_flagLivenessCache.erase(m_flagLivenessCache.begin());
    CodeImageInfo &image = m_flagLivenessCache.emplace_back();
    image.hash = hash;
    computeMicrocodeFlagLiveness(vuCode, codeSize, microAddressMask(), image.liveness.data());
    if (codeSize == kCompiledCodeSize)
        for (const auto &[compiledHash, run] : compiledPrograms())
            if (compiledHash == hash)
                image.compiled = run;
    if (!image.compiled)
        if (const char *dir = std::getenv("PS2X_VU1_DUMP_CODE"); dir && *dir)
        {
            char path[1024];
            std::snprintf(path, sizeof(path), "%s/%016llx.bin", dir, static_cast<unsigned long long>(hash));
            if (FILE *out = std::fopen(path, "wb"))
            {
                std::fwrite(vuCode, 1, codeSize, out);
                std::fclose(out);
            }
        }
    m_flagsLive = &image.liveness;
    m_compiledRun = image.compiled;
}

void VU1Interpreter::resetScheduler()
{
    m_flagPipeline = {};
    m_flagHead = 0u;
    m_flagCount = 0u;
    m_inFlightCount = 0u;
    m_inFlightHidden = false;
    m_fdiv = {};
    m_efu = {};
    resetXgkick();
    m_vfReady = {};
    m_vfReadyMax = {};
    m_viReady = {};
    m_accReady = {};
    m_efuResourceReady = 0;
    m_lastWritebackCycle = 0;
    m_nextFlagCommit = m_nextScalarCommit = 0;
    m_workingClip = m_state.clip;
    m_viBranchBackupValue = 0;
    m_viBranchBackupReg = 0;
    m_viBranchBackupValid = false;
    m_stopRequested = false;
    m_pendingHaltD = false;
    m_pendingHaltT = false;
}

void VU1Interpreter::reset()
{
    std::memset(&m_state, 0, sizeof(m_state));
    m_state.vf[0][3] = 1.0f;
    m_state.q = 1.0f;
    m_state.r = 0x3F800000u;
    m_cycle = 0;
    resetScheduler();
}

float VU1Interpreter::broadcast(const float *vf, uint8_t bc)
{
    return normalizeOperand(vf[bc & 3u]);
}

float VU1Interpreter::normalizeOperand(float value) const
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

float VU1Interpreter::normalizeResult(float value, uint32_t &laneFlags) const
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

uint32_t VU1Interpreter::microAddressMask() const
{
    return m_unit == Unit::VU1 ? 0x3FFFu : 0x0FFFu;
}

int32_t VU1Interpreter::readBranchVi(uint8_t reg) const
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

void VU1Interpreter::recordViWriteForBranch(uint8_t reg, int32_t oldValue)
{
    if (reg == 0u)
        return;
    m_viBranchBackupValue = oldValue;
    m_viBranchBackupReg = reg;
    m_viBranchBackupValid = true;
}

void VU1Interpreter::applyDest(float *dst, const float *result, uint8_t dest)
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

void VU1Interpreter::applyDestAcc(const float *result, uint8_t dest)
{
    applyDest(m_state.acc, result, dest);
}

void VU1Interpreter::normalizeFmacResult(float *result, uint8_t dest,
                                         uint8_t laneFlags[4])
{
    for (uint32_t component = 0; component < 4u; ++component)
    {
        laneFlags[component] = 0u;
        if ((dest & laneForComponent(component)) == 0u)
            continue;

        // A result with exponent 2..253 is far from zero, the denormal range and FLT_MAX: the exact
        // result has the same sign and raises no zero/underflow/overflow flag.
        uint32_t bits = 0u;
        std::memcpy(&bits, &result[component], sizeof(bits));
        const uint32_t exponent = (bits >> 23) & 0xFFu;
        if (exponent >= 2u && exponent <= 253u)
        {
            laneFlags[component] = (bits & 0x80000000u) != 0u ? 0x2u : 0u;
            continue;
        }

        double exactResult = 0.0;
        if (calculateFmacExactResult(component, exactResult))
        {
            laneFlags[component] = normalizeFmacExactResult(result[component], exactResult);
            continue;
        }

        uint32_t flags = 0u;
        result[component] = normalizeResult(result[component], flags);
        laneFlags[component] = static_cast<uint8_t>(flags);
    }
}

bool VU1Interpreter::calculateFmacExactResult(uint32_t component,
                                               double &result) const
{
    const uint32_t upper = m_currentUpperInstruction;
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t special = op >= 0x3Cu
                                ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu))
                                : 0xFFu;

    const auto vs = [this](uint32_t lane)
    {
        return static_cast<double>(m_fmacVs[lane]);
    };
    const auto vt = [this](uint32_t lane)
    {
        return static_cast<double>(m_fmacVt[lane]);
    };
    const auto acc = [this](uint32_t lane)
    {
        return static_cast<double>(m_fmacAcc[lane]);
    };

    const double q = m_fmacQ;
    const double i = m_fmacI;

    if (op < 0x3Cu)
    {
        if (op <= 0x03u)
            result = vs(component) + vt(op & 3u);
        else if (op <= 0x07u)
            result = vs(component) - vt(op & 3u);
        else if (op <= 0x0Bu)
            result = std::fma(vs(component), vt(op & 3u), acc(component));
        else if (op <= 0x0Fu)
            result = std::fma(-vs(component), vt(op & 3u), acc(component));
        else if (op >= 0x18u && op <= 0x1Bu)
            result = vs(component) * vt(op & 3u);
        else
        {
            switch (op)
            {
            case 0x1Cu:
                result = vs(component) * q;
                break;
            case 0x1Eu:
                result = vs(component) * i;
                break;
            case 0x20u:
                result = vs(component) + q;
                break;
            case 0x21u:
                result = std::fma(vs(component), q, acc(component));
                break;
            case 0x22u:
                result = vs(component) + i;
                break;
            case 0x23u:
                result = std::fma(vs(component), i, acc(component));
                break;
            case 0x24u:
                result = vs(component) - q;
                break;
            case 0x25u:
                result = std::fma(-vs(component), q, acc(component));
                break;
            case 0x26u:
                result = vs(component) - i;
                break;
            case 0x27u:
                result = std::fma(-vs(component), i, acc(component));
                break;
            case 0x28u:
                result = vs(component) + vt(component);
                break;
            case 0x29u:
                result = std::fma(vs(component), vt(component), acc(component));
                break;
            case 0x2Au:
                result = vs(component) * vt(component);
                break;
            case 0x2Cu:
                result = vs(component) - vt(component);
                break;
            case 0x2Du:
                result = std::fma(-vs(component), vt(component), acc(component));
                break;
            case 0x2Eu:
            {
                static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
                static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
                result = component == 3u
                             ? 0.0
                             : std::fma(-vs(left[component]), vt(right[component]), acc(component));
                break;
            }
            default:
                return false;
            }
        }
        return true;
    }

    if (special <= 0x03u)
        result = vs(component) + vt(special & 3u);
    else if (special <= 0x07u)
        result = vs(component) - vt(special & 3u);
    else if (special <= 0x0Bu)
        result = std::fma(vs(component), vt(special & 3u), acc(component));
    else if (special <= 0x0Fu)
        result = std::fma(-vs(component), vt(special & 3u), acc(component));
    else if (special >= 0x18u && special <= 0x1Bu)
        result = vs(component) * vt(special & 3u);
    else
    {
        switch (special)
        {
        case 0x1Cu:
            result = vs(component) * q;
            break;
        case 0x1Eu:
            result = vs(component) * i;
            break;
        case 0x20u:
            result = vs(component) + q;
            break;
        case 0x21u:
            result = std::fma(vs(component), q, acc(component));
            break;
        case 0x22u:
            result = vs(component) + i;
            break;
        case 0x23u:
            result = std::fma(vs(component), i, acc(component));
            break;
        case 0x24u:
            result = vs(component) - q;
            break;
        case 0x25u:
            result = std::fma(-vs(component), q, acc(component));
            break;
        case 0x26u:
            result = vs(component) - i;
            break;
        case 0x27u:
            result = std::fma(-vs(component), i, acc(component));
            break;
        case 0x28u:
            result = vs(component) + vt(component);
            break;
        case 0x29u:
            result = std::fma(vs(component), vt(component), acc(component));
            break;
        case 0x2Au:
            result = vs(component) * vt(component);
            break;
        case 0x2Cu:
            result = vs(component) - vt(component);
            break;
        case 0x2Du:
            result = std::fma(-vs(component), vt(component), acc(component));
            break;
        case 0x2Eu:
        {
            static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
            static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
            result = component == 3u
                         ? 0.0
                         : vs(left[component]) * vt(right[component]);
            break;
        }
        default:
            return false;
        }
    }
    return true;
}

uint8_t VU1Interpreter::normalizeFmacExactResult(float &value,
                                                  double exactResult) const
{
    const bool negative = std::signbit(exactResult);
    const double magnitude = std::fabs(exactResult);
    const double maximum = static_cast<double>(std::numeric_limits<float>::max());
    const double minimum = static_cast<double>(std::numeric_limits<float>::min());
    uint8_t flags = negative ? 0x2u : 0u;

    uint32_t bits = negative ? 0x80000000u : 0u;
    if (magnitude == 0.0L)
    {
        flags |= 0x1u;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude > maximum)
    {
        flags |= 0x8u;
        bits |= 0x7F7FFFFFu;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude < minimum)
    {
        flags |= 0x5u;
        std::memcpy(&value, &bits, sizeof(value));
    }

    return flags;
}

uint32_t VU1Interpreter::calculateFmacProductSticky(uint8_t dest) const
{
    uint32_t extraSticky = 0u;
    const uint32_t upper = m_currentUpperInstruction;
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t special = op >= 0x3Cu ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu)) : 0xFFu;
    const bool productSum =
        (op >= 0x08u && op <= 0x0Fu) ||
        op == 0x21u || op == 0x23u || op == 0x25u || op == 0x27u ||
        op == 0x29u || op == 0x2Du || op == 0x2Eu ||
        (special >= 0x08u && special <= 0x0Fu) ||
        special == 0x21u || special == 0x23u || special == 0x25u ||
        special == 0x27u || special == 0x29u || special == 0x2Du;
    if (!productSum)
        return 0u;

    {
        // Fast path: every product either has a zero factor (sticky Z) or an exponent far from the
        // underflow and overflow ranges (sticky S only, from its sign).
        static constexpr uint8_t crossLeft[4] = {1u, 2u, 0u, 3u};
        static constexpr uint8_t crossRight[4] = {2u, 0u, 1u, 3u};
        const bool broadcast = (op >= 0x08u && op <= 0x0Fu) || (special >= 0x08u && special <= 0x0Fu);
        const bool qProduct = op == 0x21u || op == 0x25u || special == 0x21u || special == 0x25u;
        const bool iProduct = op == 0x23u || op == 0x27u || special == 0x23u || special == 0x27u;
        alignas(16) float left[4];
        alignas(16) float right[4];
        for (uint32_t component = 0; component < 4u; ++component)
        {
            left[component] = m_fmacVs[op == 0x2Eu ? crossLeft[component] : component];
            right[component] = broadcast  ? m_fmacVt[(op >= 0x08u && op <= 0x0Fu ? op : special) & 3u]
                               : qProduct ? m_fmacQ
                               : iProduct ? m_fmacI
                               : op == 0x2Eu ? m_fmacVt[crossRight[component]]
                                             : m_fmacVt[component];
        }
        const __m128i leftBits = _mm_load_si128(reinterpret_cast<const __m128i *>(left));
        const __m128i rightBits = _mm_load_si128(reinterpret_cast<const __m128i *>(right));
        const __m128i leftExponent = _mm_srli_epi32(_mm_slli_epi32(leftBits, 1), 24);
        const __m128i rightExponent = _mm_srli_epi32(_mm_slli_epi32(rightBits, 1), 24);
        const __m128i zero = _mm_or_si128(_mm_cmpeq_epi32(leftExponent, _mm_setzero_si128()),
                                          _mm_cmpeq_epi32(rightExponent, _mm_setzero_si128()));
        const __m128i productExponent = _mm_sub_epi32(_mm_add_epi32(leftExponent, rightExponent), _mm_set1_epi32(127));
        const __m128i ordinary = _mm_and_si128(_mm_cmpgt_epi32(productExponent, _mm_set1_epi32(1)),
                                               _mm_cmplt_epi32(productExponent, _mm_set1_epi32(253)));
        static constexpr uint8_t kReverse4[16] = {0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE,
                                                  0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF};
        const int components = kReverse4[dest & 0xFu];
        const int zeroComponents = _mm_movemask_ps(_mm_castsi128_ps(zero)) & components;
        const int ordinaryComponents = _mm_movemask_ps(_mm_castsi128_ps(ordinary)) & components;
        if ((zeroComponents | ordinaryComponents) == components)
        {
            const int negative =
                _mm_movemask_ps(_mm_castsi128_ps(_mm_xor_si128(leftBits, rightBits))) & components;
            return (zeroComponents != 0 ? 0x1u : 0u) | (negative != 0 ? 0x2u : 0u);
        }
    }

    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((dest & laneForComponent(component)) == 0u)
            continue;
        static constexpr uint8_t crossLeft[4] = {1u, 2u, 0u, 3u};
        static constexpr uint8_t crossRight[4] = {2u, 0u, 1u, 3u};
        const uint8_t leftComponent = op == 0x2Eu ? crossLeft[component] : static_cast<uint8_t>(component);
        const float left = m_fmacVs[leftComponent];
        float right = 0.0f;
        if ((op >= 0x08u && op <= 0x0Fu) || (special >= 0x08u && special <= 0x0Fu))
        {
            right = m_fmacVt[(op >= 0x08u && op <= 0x0Fu ? op : special) & 3u];
        }
        else if (op == 0x21u || op == 0x25u || special == 0x21u || special == 0x25u)
        {
            right = m_fmacQ;
        }
        else if (op == 0x23u || op == 0x27u || special == 0x23u || special == 0x27u)
        {
            right = m_fmacI;
        }
        else if (op == 0x2Eu)
        {
            right = m_fmacVt[crossRight[component]];
        }
        else
        {
            right = m_fmacVt[component];
        }

        uint32_t leftBits = 0u, rightBits = 0u;
        std::memcpy(&leftBits, &left, sizeof(leftBits));
        std::memcpy(&rightBits, &right, sizeof(rightBits));
        const uint32_t productSign = (leftBits ^ rightBits) & 0x80000000u;
        const int32_t leftExponent = static_cast<int32_t>((leftBits >> 23) & 0xFFu);
        const int32_t rightExponent = static_cast<int32_t>((rightBits >> 23) & 0xFFu);
        if (leftExponent == 0 || rightExponent == 0)
        {
            // Normalized operands: exponent 0 is a signed zero.
            extraSticky |= 0x1u | (productSign != 0u ? 0x2u : 0u);
            continue;
        }
        // The product's exponent is leftExponent + rightExponent - 127, or one more.
        const int32_t productExponent = leftExponent + rightExponent - 127;
        if (productExponent >= 2 && productExponent + 1 <= 253)
        {
            extraSticky |= productSign != 0u ? 0x2u : 0u;
            continue;
        }

        float product = left * right;
        const double exactProduct = static_cast<double>(left) * static_cast<double>(right);
        const uint8_t productFlags = normalizeFmacExactResult(product, exactProduct);
        // Product-sum instructions report Z/S/U/O from the add/subtract result
        // as current flags, while every product condition accumulates into the
        // corresponding sticky flag.
        extraSticky |= productFlags & 0xFu;
    }
    return extraSticky;
}

void VU1Interpreter::updateFmacFlags(const uint8_t laneFlags[4], uint8_t dest,
                                     uint32_t extraSticky)
{
    if (dest == 0u)
        return;

    uint32_t mac = 0u;
    uint32_t status = 0u;
    for (uint32_t component = 0; component < 4u; ++component)
    {
        const uint8_t lane = laneForComponent(component);
        if ((dest & lane) == 0u)
            continue;

        const uint32_t flags = laneFlags[component];
        if ((flags & 0x1u) != 0u)
            mac |= lane;
        if ((flags & 0x2u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 4;
        if ((flags & 0x4u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 8;
        if ((flags & 0x8u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 12;
        status |= flags;
    }

    FlagPipelineEntry *entry = pushFlagEntry(true, 0xFFFFFFFFu);
    if (!entry)
        return;
    entry->mac = mac;
    entry->status = status;
    entry->extraSticky = extraSticky;
    entry->writesMac = true;
    entry->writesStatus = true;
}

bool VU1Interpreter::exactZeroLanes(int zeroComponents) const
{
    // A +-0 result is exact (zero flag, sign from the result) for an add or subtract, where the PS2 sum only
    // reaches zero by exact cancellation, and for a product with a zero factor, where any accumulator is
    // added to an exact zero. Other zero results may hide an underflow and take the exact path.
    if (zeroComponents == 0)
        return true;
    const uint32_t upper = m_currentUpperInstruction;
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t code = op >= 0x3Cu ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu)) : op;
    if (code <= 0x07u || code == 0x20u || code == 0x22u || code == 0x24u || code == 0x26u || code == 0x28u ||
        code == 0x2Cu)
        return true;
    const bool crossProduct = code == 0x2Eu;
    const bool broadcastProduct = (code >= 0x08u && code <= 0x0Fu) || (code >= 0x18u && code <= 0x1Bu);
    const bool qProduct = code == 0x1Cu || code == 0x21u || code == 0x25u;
    const bool iProduct = code == 0x1Eu || code == 0x23u || code == 0x27u;
    const bool laneProduct = code == 0x29u || code == 0x2Au || code == 0x2Du;
    if (!crossProduct && !broadcastProduct && !qProduct && !iProduct && !laneProduct)
        return false;
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((zeroComponents & (1 << component)) == 0)
            continue;
        static constexpr uint8_t crossLeft[4] = {1u, 2u, 0u, 3u};
        static constexpr uint8_t crossRight[4] = {2u, 0u, 1u, 3u};
        if (crossProduct && component == 3u)
            return false;
        const float left = m_fmacVs[crossProduct ? crossLeft[component] : component];
        const float right = broadcastProduct ? m_fmacVt[code & 3u]
                            : qProduct       ? m_fmacQ
                            : iProduct       ? m_fmacI
                            : crossProduct   ? m_fmacVt[crossRight[component]]
                                             : m_fmacVt[component];
        if (left != 0.0f && right != 0.0f)
            return false;
    }
    return true;
}

void VU1Interpreter::applyFmacFlags(float *result, uint8_t dest)
{
    // Destination lanes in component order (bit 0 = x) and back: DEST and the MAC flag fields put x in bit 3.
    static constexpr uint8_t kReverse4[16] = {0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE,
                                              0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF};
    const int components = kReverse4[dest & 0xFu];
    if (!m_currentFlagsLive)
    {
        if (components == 0)
            return;
        const __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i *>(result));
        const __m128i exponent = _mm_srli_epi32(_mm_and_si128(bits, _mm_set1_epi32(0x7F800000)), 23);
        const __m128i ordinary = _mm_and_si128(_mm_cmpgt_epi32(exponent, _mm_set1_epi32(1)),
                                               _mm_cmplt_epi32(exponent, _mm_set1_epi32(254)));
        const int ordinaryComponents = _mm_movemask_ps(_mm_castsi128_ps(ordinary)) & components;
        const int zeroComponents =
            _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(_mm_slli_epi32(bits, 1), _mm_setzero_si128()))) & components;
        if ((ordinaryComponents | zeroComponents) == components && exactZeroLanes(zeroComponents))
            return;
        uint8_t laneFlags[4]{};
        normalizeFmacResult(result, dest, laneFlags);
        return;
    }
    if (components != 0)
    {
        // Results with exponent 2..253 raise only the sign flag (see normalizeFmacResult).
        const __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i *>(result));
        const __m128i exponent = _mm_srli_epi32(_mm_and_si128(bits, _mm_set1_epi32(0x7F800000)), 23);
        const __m128i ordinary = _mm_and_si128(_mm_cmpgt_epi32(exponent, _mm_set1_epi32(1)),
                                               _mm_cmplt_epi32(exponent, _mm_set1_epi32(254)));
        const int ordinaryComponents = _mm_movemask_ps(_mm_castsi128_ps(ordinary)) & components;
        const int zeroComponents =
            _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(_mm_slli_epi32(bits, 1), _mm_setzero_si128()))) & components;
        if ((ordinaryComponents | zeroComponents) == components && exactZeroLanes(zeroComponents))
        {
            const int negative = _mm_movemask_ps(_mm_castsi128_ps(bits)) & components;
            const uint32_t extraSticky = calculateFmacProductSticky(dest);
            if (FlagPipelineEntry *entry = pushFlagEntry(true, 0xFFFFFFFFu))
            {
                entry->mac = (static_cast<uint32_t>(kReverse4[negative]) << 4) | kReverse4[zeroComponents];
                entry->status = (negative != 0 ? 0x2u : 0u) | (zeroComponents != 0 ? 0x1u : 0u);
                entry->extraSticky = extraSticky;
                entry->writesMac = true;
                entry->writesStatus = true;
            }
            return;
        }
    }
    uint8_t laneFlags[4]{};
    normalizeFmacResult(result, dest, laneFlags);
    updateFmacFlags(laneFlags, dest, calculateFmacProductSticky(dest));
}

void VU1Interpreter::applyFmacDest(float *dst, float *result, uint8_t dest)
{
    applyFmacFlags(result, dest);
    applyDest(dst, result, dest);
}

void VU1Interpreter::applyFmacDestAcc(float *result, uint8_t dest)
{
    applyFmacFlags(result, dest);
    applyDestAcc(result, dest);
}

void VU1Interpreter::queueFsset(uint16_t immediate)
{
    // FSSET overrides the status flags an upper instruction of the same pair would write.
    forEachFlagEntryIssuedThisCycle([](FlagPipelineEntry &entry) { entry.writesStatus = false; });
    if (FlagPipelineEntry *entry = pushFlagEntry(false, 0xFFFFFFFEu))
    {
        entry->status = static_cast<uint32_t>(immediate) & 0xFC0u;
        entry->writesSticky = true;
    }
}

void VU1Interpreter::queueClip(uint32_t clip)
{
    m_workingClip = ((m_workingClip << 6) | (clip & 0x3Fu)) & 0xFFFFFFu;
    if (FlagPipelineEntry *entry = pushFlagEntry(true, 0xFFFFFFFDu))
    {
        entry->clip = m_workingClip;
        entry->writesClip = true;
    }
}

void VU1Interpreter::queueFcset(uint32_t clip)
{
    m_workingClip = clip & 0xFFFFFFu;
    forEachFlagEntryIssuedThisCycle([](FlagPipelineEntry &entry) { entry.writesClip = false; });
    if (FlagPipelineEntry *entry = pushFlagEntry(false, 0xFFFFFFFAu))
    {
        entry->clip = m_workingClip;
        entry->writesClip = true;
    }
}

VU1Interpreter::FlagPipelineEntry *VU1Interpreter::pushFlagEntry(bool upper, uint32_t overflowCode)
{
    // Every flag write has the FMAC latency, so the queue is ordered by readiness and commits in issue order.
    if (m_flagCount == kMaxFlagEntries)
    {
        reportReservedInstruction(upper, overflowCode);
        return nullptr;
    }
    FlagPipelineEntry &entry = m_flagPipeline[(m_flagHead + m_flagCount) % kMaxFlagEntries];
    ++m_flagCount;
    entry = {};
    entry.valid = true;
    entry.issueCycle = m_cycle;
    entry.readyCycle = m_cycle + kFmacLatency;
    noteCommitCycle(m_nextFlagCommit, entry.readyCycle);
    return &entry;
}

void VU1Interpreter::queueQ(float value, uint32_t latency, uint32_t statusDi)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    m_fdiv.valid = true;
    m_fdiv.readyCycle = m_cycle + latency;
    noteCommitCycle(m_nextScalarCommit, m_fdiv.readyCycle);
    m_fdiv.value = value;
    m_fdiv.statusDi = statusDi & 0x30u;
}

void VU1Interpreter::queueP(float value, uint32_t latency)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    for (ScalarPipelineEntry &entry : m_efu)
    {
        if (!entry.valid)
        {
            entry.valid = true;
            entry.readyCycle = m_cycle + latency;
            noteCommitCycle(m_nextScalarCommit, entry.readyCycle);
            entry.value = value;
            // EFU throughput is one cycle shorter than result visibility.
            m_efuResourceReady = m_cycle + (latency > 0u ? latency - 1u : 0u);
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF9u);
}

void VU1Interpreter::queueStore(uint32_t address, const uint32_t words[4], uint8_t laneMask)
{
    // A store becomes visible at the next cycle boundary, before any later instruction or PATH1 transfer
    // reads VU memory, so writing it now is indistinguishable.
    if (!m_activeVuData || address + 16u > m_activeVuDataSize)
        return;
    catchUpXgkick();
    if (m_cycle + 1u > m_lastWritebackCycle)
        m_lastWritebackCycle = m_cycle + 1u;
    uint32_t *target = reinterpret_cast<uint32_t *>(m_activeVuData + address);
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((laneMask & laneForComponent(component)) != 0u)
            target[component] = words[component];
    }
}

void VU1Interpreter::commitReadyPipelines()
{
    // Register and memory results are written at issue (see run()); only flags, Q and P are delayed.
    if (m_cycle < m_nextFlagCommit && m_cycle < m_nextScalarCommit)
        return;

    if (m_cycle >= m_nextFlagCommit)
    {
        while (m_flagCount != 0u && m_flagPipeline[m_flagHead].readyCycle <= m_cycle)
        {
            FlagPipelineEntry &entry = m_flagPipeline[m_flagHead];
            if (entry.writesMac)
                m_state.mac = entry.mac;
            if (entry.writesStatus)
            {
                const uint32_t current = entry.status & 0xFu;
                m_state.status = (m_state.status & 0xFF0u) | current | ((current | entry.extraSticky) << 6);
            }
            if (entry.writesSticky)
            {
                m_state.status = (m_state.status & 0x03Fu) | (entry.status & 0xFC0u);
            }
            if (entry.writesClip)
                m_state.clip = entry.clip;
            entry = {};
            m_flagHead = (m_flagHead + 1u) % kMaxFlagEntries;
            --m_flagCount;
        }
        m_nextFlagCommit = m_flagCount != 0u ? m_flagPipeline[m_flagHead].readyCycle : UINT64_MAX;
    }

    if (m_cycle >= m_nextScalarCommit)
    {
        m_nextScalarCommit = UINT64_MAX;
        if (m_fdiv.valid && m_fdiv.readyCycle <= m_cycle)
        {
            m_state.q = m_fdiv.value;
            const uint32_t currentDi = m_fdiv.statusDi & 0x30u;
            m_state.status = (m_state.status & 0xFCFu) | currentDi | (currentDi << 6);
            m_fdiv = {};
        }
        else if (m_fdiv.valid)
        {
            noteCommitCycle(m_nextScalarCommit, m_fdiv.readyCycle);
        }

        for (ScalarPipelineEntry &entry : m_efu)
        {
            if (!entry.valid)
                continue;
            if (entry.readyCycle <= m_cycle)
            {
                m_state.p = entry.value;
                entry = {};
            }
            else
            {
                noteCommitCycle(m_nextScalarCommit, entry.readyCycle);
            }
        }
    }

}

void VU1Interpreter::progressXgkick(uint64_t cycles)
{
    if (!m_xgkick.active || !m_activeVuData || m_activeVuDataSize == 0u)
        return;

    m_xgkick.cycleCredit += static_cast<uint32_t>(std::min<uint64_t>(cycles, 0x40000000u));
    while (m_xgkick.active && m_xgkick.cycleCredit >= 2u)
    {
        m_xgkick.cycleCredit -= 2u;
        if (m_xgkick.copiedBytes > XgkickPipeline::kBufferSize - 16u)
        {
            reportReservedInstruction(false, 0xFFFFFFFBu);
            m_xgkick.active = false;
            return;
        }

        const uint32_t qwordOffset = m_xgkick.copiedBytes;
        const uint32_t source = (m_xgkick.sourceAddress + m_xgkick.copiedBytes) % m_activeVuDataSize;
        if (source + 16u <= m_activeVuDataSize)
        {
            std::memcpy(m_xgkick.packet.data() + qwordOffset, m_activeVuData + source, 16u);
        }
        else
        {
            for (uint32_t i = 0; i < 16u; ++i)
                m_xgkick.packet[qwordOffset + i] = m_activeVuData[(source + i) % m_activeVuDataSize];
        }
        m_xgkick.copiedBytes += 16u;

        if (m_xgkick.currentTagEnd == 0u)
        {
            uint64_t tagLo = 0;
            std::memcpy(&tagLo, m_xgkick.packet.data() + qwordOffset, sizeof(tagLo));
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint32_t format = static_cast<uint32_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;

            uint64_t tagBytes = 16u;
            if (format == 0u)
                tagBytes += static_cast<uint64_t>(nloop) * nreg * 16u;
            else if (format == 1u)
                tagBytes += ((static_cast<uint64_t>(nloop) * nreg + 1u) & ~1ull) * 8u;
            else // IMAGE; FLG 3 ("disable") behaves as IMAGE on the GS
                tagBytes += static_cast<uint64_t>(nloop) * 16u;

            if (tagBytes > XgkickPipeline::kBufferSize - qwordOffset)
            {
                reportReservedInstruction(false, 0xFFFFFFFBu);
                m_xgkick.active = false;
                return;
            }
            m_xgkick.currentTagEnd = qwordOffset + static_cast<uint32_t>(tagBytes);
            m_xgkick.currentTagEop = ((tagLo >> 15) & 1u) != 0u;
            if (m_xgkick.currentTagEop)
                m_xgkick.totalBytes = m_xgkick.currentTagEnd;
        }

        if (m_xgkick.copiedBytes >= m_xgkick.currentTagEnd)
        {
            if (m_xgkick.currentTagEop)
                finishXgkick();
            else
            {
                // The next transferred qword is another GIFtag.
                m_xgkick.currentTagEnd = 0u;
                m_xgkick.currentTagEop = false;
            }
        }
    }
}

void VU1Interpreter::finishXgkick()
{
    if (!m_xgkick.active)
        return;

    if (m_packetObserver)
        m_packetObserver(m_xgkick.packet.data(), m_xgkick.totalBytes);
    if (m_activeMemory)
        m_activeMemory->submitGifPacket(GifPathId::Path1, m_xgkick.packet.data(), m_xgkick.totalBytes);
    else if (m_activeGs)
        m_activeGs->processGIFPacket(m_xgkick.packet.data(), m_xgkick.totalBytes);
    m_xgkick.active = false;
}

void VU1Interpreter::resetXgkick()
{
    // Everything but the packet buffer, which is only read up to copiedBytes.
    m_xgkick.sourceAddress = 0u;
    m_xgkick.totalBytes = 0u;
    m_xgkick.copiedBytes = 0u;
    m_xgkick.currentTagEnd = 0u;
    m_xgkick.cycleCredit = 0u;
    m_xgkick.issueCycle = 0u;
    m_xgkick.active = false;
    m_xgkick.currentTagEop = false;
    m_xgkickBacklog = 0u;
}

void VU1Interpreter::startXgkick(uint32_t qwordAddress)
{
    if (m_unit != Unit::VU1 || !m_activeVuData || m_activeVuDataSize < 16u)
        return;

    const uint32_t sourceAddress = (qwordAddress * 16u) % m_activeVuDataSize;
    resetXgkick();
    m_xgkick.active = true;
    m_xgkick.sourceAddress = sourceAddress;
    m_xgkick.cycleCredit = 1u; // XGKICK's issue cycle counts toward PATH1.
    m_xgkick.issueCycle = m_cycle;
}



bool VU1Interpreter::pipelinesPending() const
{
    if (m_fdiv.valid || m_xgkick.active || m_cycle < m_lastWritebackCycle)
        return true;
    for (const ScalarPipelineEntry &entry : m_efu)
        if (entry.valid)
            return true;
    if (m_flagCount != 0u)
        return true;
    return false;
}

void VU1Interpreter::flushPipelines()
{
    for (catchUpXgkick(); pipelinesPending(); catchUpXgkick())
        advanceOneCycle();
}



VU1Interpreter::InstructionUsage VU1Interpreter::decodeUpperUsage(uint32_t upper) const
{
    InstructionUsage usage;
    usage.pipeline = PipelineFmac;
    usage.latency = kFmacLatency;

    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t dest = DEST(upper);
    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);
    const uint8_t fd = FD(upper);

    if (op <= 0x2Fu)
    {
        addVfRead(usage, fs, dest);
        addVfWrite(usage, fd, dest);
        if (op <= 0x1Bu)
            addVfRead(usage, ft, laneForComponent(op & 3u));
        else if (op >= 0x28u)
            addVfRead(usage, ft, op == 0x2Eu ? 0xEu : dest);
        if (op == 0x08u || op == 0x09u || op == 0x0Au || op == 0x0Bu ||
            op == 0x0Cu || op == 0x0Du || op == 0x0Eu || op == 0x0Fu ||
            op == 0x21u || op == 0x23u || op == 0x25u || op == 0x27u ||
            op == 0x29u || op == 0x2Du || op == 0x2Eu)
        {
            usage.accRead = dest;
        }
        return usage;
    }

    if (op >= 0x3Cu)
    {
        const uint8_t special = static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu));
        const bool writesAcc =
            special <= 0x0Fu ||
            (special >= 0x18u && special <= 0x1Cu) ||
            special == 0x1Eu ||
            (special >= 0x20u && special <= 0x2Au) ||
            (special >= 0x2Cu && special <= 0x2Eu);
        if (writesAcc)
        {
            addVfRead(usage, fs, dest);
            if (special <= 0x1Bu)
                addVfRead(usage, ft, laneForComponent(special & 3u));
            else if ((special >= 0x28u && special <= 0x2Eu))
                addVfRead(usage, ft, special == 0x2Eu ? 0xEu : dest);
            usage.accWrite = dest;
            if ((special >= 0x08u && special <= 0x0Fu) ||
                special == 0x21u || special == 0x23u || special == 0x25u ||
                special == 0x27u || special == 0x29u || special == 0x2Du)
            {
                usage.accRead = dest;
            }
        }
        else if (special >= 0x10u && special <= 0x17u)
        {
            addVfRead(usage, fs, dest);
            addVfWrite(usage, ft, dest);
        }
        else if (special == 0x1Du)
        {
            addVfRead(usage, fs, dest);
            addVfWrite(usage, ft, dest);
        }
        else if (special == 0x1Fu)
        {
            addVfRead(usage, fs, 0xEu);
            addVfRead(usage, ft, 0x1u);
            usage.writesClip = true;
        }
        else if (special != 0x2Fu && special != 0x30u)
        {
            usage.reserved = true;
        }
        return usage;
    }

    usage.reserved = true;
    return usage;
}

VU1Interpreter::InstructionUsage VU1Interpreter::decodeLowerUsage(uint32_t lower) const
{
    InstructionUsage usage;
    if (lower == 0u || lower == 0x8000033Cu)
        return usage;

    const uint8_t opHi = static_cast<uint8_t>((lower >> 25) & 0x7Fu);
    const uint8_t vfT = FT(lower);
    const uint8_t vfS = FS(lower);
    const uint8_t viT = VIT(lower);
    const uint8_t viS = VIS(lower);
    const uint8_t viD = VID(lower);
    const uint8_t dest = DEST(lower);
    auto readVi = [&](uint8_t reg)
    {
        if (reg != 0u)
            usage.viRead |= static_cast<uint16_t>(1u << reg);
    };
    auto writeVi = [&](uint8_t reg)
    {
        if (reg != 0u)
            usage.viWrite |= static_cast<uint16_t>(1u << reg);
    };

    switch (opHi)
    {
    case 0x00:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        addVfWrite(usage, vfT, dest);
        return usage;
    case 0x01:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viT);
        addVfRead(usage, vfS, dest);
        return usage;
    case 0x04:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x05:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viS);
        readVi(viT);
        return usage;
    case 0x08:
    case 0x09:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x10:
    case 0x12:
    case 0x13:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.readsClip = true;
        writeVi(1u);
        return usage;
    case 0x11:
        usage.pipeline = PipelineFmac;
        usage.latency = kFmacLatency;
        usage.writesClip = true;
        return usage;
    case 0x14:
    case 0x16:
    case 0x17:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        writeVi(viT);
        return usage;
    case 0x15:
        usage.pipeline = PipelineFmac;
        usage.latency = kFmacLatency;
        return usage;
    case 0x18:
    case 0x1A:
    case 0x1B:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x1C:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.readsClip = true;
        writeVi(viT);
        return usage;
    case 0x20:
        usage.pipeline = PipelineBranch;
        return usage;
    case 0x21:
        usage.pipeline = PipelineBranch;
        usage.latency = 1u;
        writeVi(viT);
        return usage;
    case 0x24:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        return usage;
    case 0x25:
        usage.pipeline = PipelineBranch;
        usage.latency = 1u;
        readVi(viS);
        writeVi(viT);
        return usage;
    case 0x28:
    case 0x29:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        readVi(viT);
        return usage;
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x2F:
        usage.pipeline = PipelineBranch;
        readVi(viS);
        return usage;
    case 0x40:
        break;
    default:
        usage.reserved = true;
        return usage;
    }

    const uint8_t direct = static_cast<uint8_t>(lower & 0x3Fu);
    if (direct == 0x30u || direct == 0x31u || direct == 0x34u || direct == 0x35u)
    {
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        readVi(viT);
        writeVi(viD);
        return usage;
    }
    if (direct == 0x32u)
    {
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viT);
        return usage;
    }
    if (direct < 0x3Cu)
    {
        usage.reserved = true;
        return usage;
    }

    const uint8_t special = static_cast<uint8_t>((lower & 3u) | ((lower >> 4) & 0x7Cu));
    switch (special)
    {
    case 0x30:
    case 0x31:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfRead(usage, vfS, special == 0x31u ? 0xFu : dest);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x34:
    case 0x36:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        usage.viLatency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viS);
        writeVi(viS);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x35:
    case 0x37:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        readVi(viT);
        writeVi(viT);
        addVfRead(usage, vfS, dest);
        break;
    case 0x38:
        usage.pipeline = PipelineFdiv;
        usage.latency = 7u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x39:
        usage.pipeline = PipelineFdiv;
        usage.latency = 7u;
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x3A:
        usage.pipeline = PipelineFdiv;
        usage.latency = 13u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        addVfRead(usage, vfT, laneForComponent((lower >> 23) & 3u));
        break;
    case 0x3B:
        usage.pipeline = PipelineFdiv;
        usage.waitQ = true;
        break;
    case 0x3C:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        usage.delaysNextBranchRead = true;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        writeVi(viT);
        break;
    case 0x3D:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        readVi(viS);
        addVfWrite(usage, vfT, dest);
        break;
    case 0x3E:
        usage.pipeline = PipelineLsu;
        usage.latency = 4u;
        readVi(viS);
        writeVi(viT);
        break;
    case 0x3F:
        usage.pipeline = PipelineLsu;
        usage.latency = 1u;
        readVi(viS);
        readVi(viT);
        break;
    case 0x40:
    case 0x41:
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfWrite(usage, vfT, dest);
        break;
    case 0x42:
    case 0x43:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        break;
    case 0x64:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineFmac;
        usage.latency = 4u;
        addVfWrite(usage, vfT, dest);
        break;
    case 0x68:
    case 0x69:
        usage.pipeline = PipelineIalu;
        usage.latency = 1u;
        writeVi(viT);
        break;
    case 0x6C:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineXgkick;
        usage.latency = 2u;
        readVi(viS);
        break;
    case 0x70:
    case 0x71:
    case 0x72:
    case 0x73:
    case 0x74:
    case 0x75:
    case 0x76:
    case 0x77:
    case 0x78:
    case 0x79:
    case 0x7A:
    case 0x7C:
    case 0x7D:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineEfu;
        switch (special)
        {
        case 0x70:
            usage.latency = 11u;
            break;
        case 0x71:
        case 0x72:
        case 0x77:
            usage.latency = 18u;
            break;
        case 0x73:
            usage.latency = 24u;
            break;
        case 0x74:
        case 0x75:
        case 0x7C:
            usage.latency = 54u;
            break;
        case 0x76:
        case 0x78:
        case 0x7A:
            usage.latency = 12u;
            break;
        case 0x79:
            usage.latency = 29u;
            break;
        case 0x7D:
            usage.latency = 44u;
            break;
        default:
            break;
        }
        if (special >= 0x70u && special <= 0x73u)
            addVfRead(usage, vfS, 0xEu);
        else if (special == 0x74u)
            addVfRead(usage, vfS, 0xCu);
        else if (special == 0x75u)
            addVfRead(usage, vfS, 0xAu);
        else if (special == 0x76u)
            addVfRead(usage, vfS, 0xFu);
        else
            addVfRead(usage, vfS, laneForComponent((lower >> 21) & 3u));
        break;
    case 0x7B:
        if (m_unit == Unit::VU0)
        {
            usage.reserved = true;
            break;
        }
        usage.pipeline = PipelineEfu;
        usage.waitP = true;
        break;
    default:
        usage.reserved = true;
        break;
    }
    return usage;
}

VU1Interpreter::DecodedInstructionPair VU1Interpreter::decodeInstructionPair(const uint8_t *vuCode, uint32_t pc) const
{
    DecodedInstructionPair decoded;
    std::memcpy(&decoded.lower, vuCode + pc, sizeof(decoded.lower));
    std::memcpy(&decoded.upper, vuCode + pc + sizeof(decoded.lower), sizeof(decoded.upper));
    decoded.iBit = (decoded.upper & 0x80000000u) != 0u;
    decoded.eBit = (decoded.upper & 0x40000000u) != 0u;
    decoded.mBit = (decoded.upper & 0x20000000u) != 0u;
    decoded.dBit = (decoded.upper & 0x10000000u) != 0u;
    decoded.tBit = (decoded.upper & 0x08000000u) != 0u;
    decoded.upperUsage = decodeUpperUsage(decoded.upper);
    decoded.upperHandler = selectUpperHandler(decoded.upper);
    if (!decoded.iBit)
        decoded.lowerUsage = decodeLowerUsage(decoded.lower);

    const uint8_t upperWriteReg = decoded.upperUsage.vfWrite.reg;
    if (upperWriteReg != 0u && (vfReadLanes(decoded.lowerUsage, upperWriteReg) != 0u || decoded.lowerUsage.vfWrite.reg == upperWriteReg))
    {
        decoded.upperVfShadowReg = upperWriteReg;
        if (decoded.lowerUsage.vfWrite.reg == upperWriteReg)
            decoded.suppressedLowerVf = upperWriteReg;
    }
    if (const uint32_t viWrites = decoded.lowerUsage.viWrite & 0xFFFEu; viWrites != 0u)
        decoded.writtenVi = static_cast<uint8_t>(_tzcnt_u32(viWrites));
    return decoded;
}

const VU1Interpreter::DecodedInstructionPair &VU1Interpreter::getDecodedInstructionPairForPc(
    const uint8_t *vuCode, uint32_t codeSize, PS2Memory *memory, uint32_t pc)
{
    // Without a PS2Memory the caller owns the code buffer and reports changes via invalidateDecodedCode().
    const bool trackedVuCode = memory == nullptr ||
                               (m_unit == Unit::VU1 && vuCode == memory->getVU1Code()) ||
                               (m_unit == Unit::VU0 && vuCode == memory->getVU0Code());
    const uint32_t pairIndex = pc / 8u;
    if ((pc & 7u) != 0u || !trackedVuCode || pairIndex >= kMaxDecodedPairs)
    {
        m_uncachedDecodedPair = decodeInstructionPair(vuCode, pc);
        return m_uncachedDecodedPair;
    }

    // Pairs are decoded on first use and stamped with the code generation they were decoded from, so a
    // microcode upload only costs the pairs that actually run afterwards.
    if (m_cachedVuCode != vuCode || m_cachedMemory != memory || m_cachedCodeSize != codeSize)
    {
        m_decodedPairStamp.fill(0u);
        m_cachedVuCode = vuCode;
        m_cachedMemory = memory;
        m_cachedCodeSize = codeSize;
    }
    const uint64_t stamp = memory == nullptr ? m_externalCodeGeneration + 1u
                           : (m_unit == Unit::VU1 ? memory->getVU1CodeGeneration() : memory->getVU0CodeGeneration()) + 1u;
    if (m_decodedPairStamp[pairIndex] != stamp)
    {
        m_decodedCodeCache[pairIndex] = decodeInstructionPair(vuCode, pc);
        m_decodedPairStamp[pairIndex] = stamp;
    }
    return m_decodedCodeCache[pairIndex];
}

void VU1Interpreter::reportReservedInstruction(bool upper, uint32_t instruction)
{
    RUNTIME_ERROR(
        "[VU" << (m_unit == Unit::VU1 ? "1" : "0")
              << " reserved " << (upper ? "upper" : "lower")
              << "] cycle=" << m_cycle
              << " pc=0x" << std::hex << m_state.pc
              << " instruction=0x" << instruction
              << std::dec << '\n');
    m_stopRequested = true;
}

void VU1Interpreter::logInFlightWrite(InFlightWrite::Kind kind, uint8_t reg, uint8_t lanes, uint64_t readyCycle,
                                      const float oldValue[4], const float newValue[4])
{
    // Writes become ready at most a few cycles after issue, so the oldest entry of the ring has
    // always landed when it is overwritten.
    InFlightWrite &entry = m_inFlight[m_inFlightNext];
    m_inFlightNext = (m_inFlightNext + 1u) % kInFlightWrites;
    if (m_inFlightCount < kInFlightWrites)
        ++m_inFlightCount;
    entry.kind = kind;
    entry.reg = reg;
    entry.lanes = lanes;
    entry.readyCycle = readyCycle;
    std::memcpy(entry.oldValue, oldValue, sizeof(entry.oldValue));
    std::memcpy(entry.newValue, newValue, sizeof(entry.newValue));
}

void VU1Interpreter::applyInFlightWrite(const InFlightWrite &entry, bool newValue)
{
    const float *value = newValue ? entry.newValue : entry.oldValue;
    if (entry.kind == InFlightWrite::Kind::Vi)
    {
        m_state.vi[entry.reg] = static_cast<int32_t>(value[0]);
        return;
    }
    float *target = entry.kind == InFlightWrite::Kind::Acc ? m_state.acc : m_state.vf[entry.reg];
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((entry.lanes & laneForComponent(component)) != 0u)
            target[component] = value[component];
    }
}

void VU1Interpreter::hideInFlightWrites()
{
    // The run stopped before these results' writeback: show the registers as the pipeline would,
    // undoing the newest writes first.
    for (uint32_t back = 1u; back <= m_inFlightCount; ++back)
    {
        const InFlightWrite &entry = m_inFlight[(m_inFlightNext + kInFlightWrites - back) % kInFlightWrites];
        if (entry.readyCycle > m_cycle)
        {
            applyInFlightWrite(entry, false);
            m_inFlightHidden = true;
        }
    }
}

void VU1Interpreter::revealInFlightWrites()
{
    if (!m_inFlightHidden)
        return;
    m_inFlightHidden = false;
    for (uint32_t forward = m_inFlightCount; forward > 0u; --forward)
    {
        const InFlightWrite &entry = m_inFlight[(m_inFlightNext + kInFlightWrites - forward) % kInFlightWrites];
        if (entry.readyCycle > m_cycle) // hidden at the end of the previous run (the clock has not moved)
            applyInFlightWrite(entry, true);
    }
}

void VU1Interpreter::execute(uint8_t *vuCode, uint32_t codeSize,
                             uint8_t *vuData, uint32_t dataSize,
                             GS &gs, PS2Memory *memory,
                             uint32_t startPC, uint32_t top, uint32_t itop,
                             uint32_t maxCycles)
{
    resetScheduler();
    m_state.pc = startPC & microAddressMask();
    m_state.ebit = false;
    m_state.haltAfterDelaySlot = false;
    m_state.stoppedByD = false;
    m_state.stoppedByT = false;
    m_state.top = top;
    m_state.itop = itop;
    m_state.branchPending = false;
    m_state.branchTarget = 0;
    m_state.branchDelay = 0;
    m_state.vf[0][0] = 0.0f;
    m_state.vf[0][1] = 0.0f;
    m_state.vf[0][2] = 0.0f;
    m_state.vf[0][3] = 1.0f;
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

void VU1Interpreter::resume(uint8_t *vuCode, uint32_t codeSize,
                            uint8_t *vuData, uint32_t dataSize,
                            GS &gs, PS2Memory *memory,
                            uint32_t top, uint32_t itop, uint32_t maxCycles)
{
    m_state.top = top;
    m_state.itop = itop;
    m_state.stoppedByD = false;
    m_state.stoppedByT = false;
    run(vuCode, codeSize, vuData, dataSize, gs, memory, maxCycles);
}

void VU1Interpreter::run(uint8_t *vuCode, uint32_t codeSize,
                         uint8_t *vuData, uint32_t dataSize,
                         GS &gs, PS2Memory *memory, uint32_t maxCycles)
{
    m_activeVuData = vuData;
    m_activeVuDataSize = dataSize;
    m_activeGs = &gs;
    m_activeMemory = memory;

    const int previousRoundingMode = std::fegetround();
    const bool useVuRounding = std::fesetround(FE_TOWARDZERO) == 0;
    const uint64_t budgetEnd = m_cycle + maxCycles;
    bool programEnded = false;
    commitReadyPipelines(); // afterwards every clock advance commits what became ready
    revealInFlightWrites();
    // Only short time slices can stop with writes in flight that a caller then inspects; run-to-completion
    // budgets (VIF MSCAL/MSCNT) skip the logging.
    m_logInFlight = maxCycles < (1u << 20);
    if (!m_logInFlight)
        m_inFlightCount = 0u;
    // The decode cache stamp only changes when microcode is uploaded, which cannot happen during a run.
    uint64_t decodeStamp = 0u;
    if (m_cachedVuCode == vuCode && m_cachedMemory == memory && m_cachedCodeSize == codeSize)
    {
        if (memory == nullptr)
            decodeStamp = m_externalCodeGeneration + 1u;
        else if (m_unit == Unit::VU1 && vuCode == memory->getVU1Code())
            decodeStamp = memory->getVU1CodeGeneration() + 1u;
        else if (m_unit == Unit::VU0 && vuCode == memory->getVU0Code())
            decodeStamp = memory->getVU0CodeGeneration() + 1u;
    }
    if (m_elideDeadFlags && decodeStamp != 0u && decodeStamp != m_flagsLiveStamp)
        refreshFlagLiveness(vuCode, codeSize, decodeStamp);
    const FlagLiveness *flagsLive = m_elideDeadFlags && decodeStamp != 0u ? m_flagsLive : nullptr;
    const CompiledRun compiled = flagsLive ? m_compiledRun : nullptr;
    while (m_cycle < budgetEnd && !m_stopRequested)
    {
        if (compiled)
        {
            const StepResult step = compiled(*this, budgetEnd, vuData, dataSize, gs, memory);
            if (step == StepResult::Stop)
                break;
            if (step == StepResult::Ended)
            {
                programEnded = true;
                break;
            }
            if (m_cycle >= budgetEnd || m_stopRequested)
                break;
        }
        if (m_state.pc + 8u > codeSize)
            break;

        const uint32_t pairIndex = m_state.pc / 8u;
        const bool flagsLiveHere = flagsLive == nullptr || (m_state.pc & 7u) != 0u || pairIndex >= kMaxDecodedPairs ||
                                   (*flagsLive)[pairIndex];
        const DecodedInstructionPair &decoded =
            (decodeStamp != 0u && (m_state.pc & 7u) == 0u && pairIndex < kMaxDecodedPairs &&
             m_decodedPairStamp[pairIndex] == decodeStamp)
                ? m_decodedCodeCache[pairIndex]
                : getDecodedInstructionPairForPc(vuCode, codeSize, memory, m_state.pc);
        m_currentFlagsLive = flagsLiveHere;
        const StepResult step = stepPair(
            decoded, budgetEnd, codeSize, [this, &decoded]() { (this->*decoded.upperHandler)(decoded.upper); },
            [&]() { execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper); });
        if (step == StepResult::Stop)
            break;
        if (step == StepResult::Ended)
        {
            programEnded = true;
            break;
        }
    }

    catchUpXgkick();
    if (!programEnded)
        hideInFlightWrites();

    // Small budgets are deliberate time slices; a run-to-completion budget that expires means a
    // runaway program.
    if (!programEnded && m_cycle >= budgetEnd && maxCycles >= (1u << 20))
    {
        static uint32_t s_budgetWarnings = 0u;
        if (s_budgetWarnings < 8u)
        {
            ++s_budgetWarnings;
            std::fprintf(stderr, "[VU%u] program still running after %u cycles at pc=0x%x\n",
                         m_unit == Unit::VU1 ? 1u : 0u, maxCycles, m_state.pc);
        }
    }
    if (programEnded)
    {
        flushPipelines();
        m_state.ebit = false;
        m_state.haltAfterDelaySlot = false;
        m_pendingHaltD = false;
        m_pendingHaltT = false;
    }
    m_state.cycles = m_cycle;
    m_currentFlagsLive = true;
    if (useVuRounding && previousRoundingMode != -1)
        std::fesetround(previousRoundingMode);
}
