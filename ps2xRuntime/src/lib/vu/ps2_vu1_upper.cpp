#include "runtime/ps2_vu1.h"
#include "ps2_vu1_detail.h"

#include <cmath>
#include <emmintrin.h>
#include <cstring>
#include <limits>

namespace
{
    int32_t vuFloatToInt(float value, float scale)
    {
        const double scaled = static_cast<double>(value) * static_cast<double>(scale);
        if (scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()))
            return std::numeric_limits<int32_t>::max();
        if (scaled <= static_cast<double>(std::numeric_limits<int32_t>::min()))
            return std::numeric_limits<int32_t>::min();
        return static_cast<int32_t>(scaled);
    }

    // MAX/MINI compare raw sign-magnitude bit patterns and pass the chosen operand through unchanged, so
    // denormals (often integer data, e.g. colors unpacked by VIF) and exponent-255 values survive.
    int64_t vuMinMaxKey(float value)
    {
        uint32_t bits = 0u;
        std::memcpy(&bits, &value, sizeof(bits));
        const int64_t magnitude = static_cast<int64_t>(bits & 0x7FFFFFFFu);
        return (bits & 0x80000000u) != 0u ? -magnitude : magnitude;
    }

    float vuMax(float a, float b) { return vuMinMaxKey(a) > vuMinMaxKey(b) ? a : b; }
    float vuMini(float a, float b) { return vuMinMaxKey(a) < vuMinMaxKey(b) ? a : b; }

    // normalizeOperand() on four lanes: denormals become signed zero, exponent 255 becomes +-FLT_MAX.
    void normalizeVector(const float *source, float *target)
    {
        const __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i *>(source));
        const __m128i exponentMask = _mm_set1_epi32(0x7F800000);
        const __m128i sign = _mm_and_si128(bits, _mm_set1_epi32(static_cast<int>(0x80000000u)));
        const __m128i exponent = _mm_and_si128(bits, exponentMask);
        const __m128i zeroExponent = _mm_cmpeq_epi32(exponent, _mm_setzero_si128());
        const __m128i maxExponent = _mm_cmpeq_epi32(exponent, exponentMask);
        __m128i result = _mm_or_si128(_mm_andnot_si128(zeroExponent, bits), _mm_and_si128(zeroExponent, sign));
        result = _mm_or_si128(_mm_andnot_si128(maxExponent, result),
                              _mm_and_si128(maxExponent, _mm_or_si128(sign, _mm_set1_epi32(0x7F7FFFFF))));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(target), result);
    }
}

// ============================================================================
// Upper instructions (FMAC pipeline)
// ============================================================================
void VU1Interpreter::execUpper(uint32_t instr)
{
    m_currentUpperInstruction = instr;
    uint8_t dest = DEST(instr);
    uint8_t ft = FT(instr);
    uint8_t fs = FS(instr);
    uint8_t fd = FD(instr);
    uint8_t op = instr & 0x3F;

    // NOP (special 0x2F/0x30) is the most common upper instruction.
    const uint32_t low11 = instr & 0x7FFu;
    if (low11 == 0x2FFu || low11 == 0x33Cu)
        return;

    float *vd = m_state.vf[fd];
    alignas(16) float normalizedVs[4];
    alignas(16) float normalizedVt[4];
    alignas(16) float normalizedAcc[4];
    normalizeVector(m_state.vf[fs], normalizedVs);
    normalizeVector(m_state.vf[ft], normalizedVt);
    normalizeVector(m_state.acc, normalizedAcc);
    const float *vs = normalizedVs;
    const float *vt = normalizedVt;
    const float *acc = normalizedAcc;
    const float q = normalizeOperand(m_state.q);
    const float i = normalizeOperand(m_state.i);
    float result[4];

    // Upper opcode decoding (bits 5:0 of upper word)
    switch (op)
    {
    case 0x00:
    case 0x01:
    case 0x02:
    case 0x03: // ADDbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + bc;
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x04:
    case 0x05:
    case 0x06:
    case 0x07: // SUBbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - bc;
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x08:
    case 0x09:
    case 0x0A:
    case 0x0B: // MADDbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * bc;
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x0C:
    case 0x0D:
    case 0x0E:
    case 0x0F: // MSUBbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * bc;
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x13: // MAXbc
    {
        const float bc = m_state.vf[ft][op & 3];
        for (int c = 0; c < 4; c++)
            result[c] = vuMax(m_state.vf[fs][c], bc);
        applyDest(vd, result, dest);
        return;
    }
    case 0x14:
    case 0x15:
    case 0x16:
    case 0x17: // MINIbc
    {
        const float bc = m_state.vf[ft][op & 3];
        for (int c = 0; c < 4; c++)
            result[c] = vuMini(m_state.vf[fs][c], bc);
        applyDest(vd, result, dest);
        return;
    }
    case 0x18:
    case 0x19:
    case 0x1A:
    case 0x1B: // MULbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * bc;
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x1C: // MULq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * q;
        applyFmacDest(vd, result, dest);
        return;
    case 0x1D: // MAXi
        for (int c = 0; c < 4; c++)
            result[c] = vuMax(m_state.vf[fs][c], m_state.i);
        applyDest(vd, result, dest);
        return;
    case 0x1E: // MULi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * i;
        applyFmacDest(vd, result, dest);
        return;
    case 0x1F: // MINIi
        for (int c = 0; c < 4; c++)
            result[c] = vuMini(m_state.vf[fs][c], m_state.i);
        applyDest(vd, result, dest);
        return;
    case 0x20: // ADDq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + q;
        applyFmacDest(vd, result, dest);
        return;
    case 0x21: // MADDq
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * q;
        applyFmacDest(vd, result, dest);
        return;
    case 0x22: // ADDi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + i;
        applyFmacDest(vd, result, dest);
        return;
    case 0x23: // MADDi
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * i;
        applyFmacDest(vd, result, dest);
        return;
    case 0x24: // SUBq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - q;
        applyFmacDest(vd, result, dest);
        return;
    case 0x25: // MSUBq
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * q;
        applyFmacDest(vd, result, dest);
        return;
    case 0x26: // SUBi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - i;
        applyFmacDest(vd, result, dest);
        return;
    case 0x27: // MSUBi
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * i;
        applyFmacDest(vd, result, dest);
        return;
    case 0x28: // ADD
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + vt[c];
        applyFmacDest(vd, result, dest);
        return;
    case 0x29: // MADD
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * vt[c];
        applyFmacDest(vd, result, dest);
        return;
    case 0x2A: // MUL
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * vt[c];
        applyFmacDest(vd, result, dest);
        return;
    case 0x2B: // MAX
        for (int c = 0; c < 4; c++)
            result[c] = vuMax(m_state.vf[fs][c], m_state.vf[ft][c]);
        applyDest(vd, result, dest);
        return;
    case 0x2C: // SUB
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - vt[c];
        applyFmacDest(vd, result, dest);
        return;
    case 0x2D: // MSUB
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * vt[c];
        applyFmacDest(vd, result, dest);
        return;
    case 0x2E: // OPMSUB
        result[0] = acc[0] - vs[1] * vt[2];
        result[1] = acc[1] - vs[2] * vt[0];
        result[2] = acc[2] - vs[0] * vt[1];
        result[3] = 0.0f;
        applyFmacDest(vd, result, dest);
        return;
    case 0x2F: // MINI
        for (int c = 0; c < 4; c++)
            result[c] = vuMini(m_state.vf[fs][c], m_state.vf[ft][c]);
        applyDest(vd, result, dest);
        return;

    // Upper special group (low op 0x3C..0x3F).
    // Like lower1 special, the real selector is not just bits 5:0.  Dobie decodes:
    //   op = (instr & 0x3) | ((instr >> 4) & 0x7C)
    // Several instructions in this group also use FT as the destination, not FD.
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    {
        const uint8_t specialOp = static_cast<uint8_t>((instr & 0x3u) | ((instr >> 4) & 0x7Cu));
        float *vtDest = m_state.vf[ft];

        switch (specialOp)
        {
        case 0x00:
        case 0x01:
        case 0x02:
        case 0x03: // ADDAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + bc;
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x04:
        case 0x05:
        case 0x06:
        case 0x07: // SUBAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - bc;
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x08:
        case 0x09:
        case 0x0A:
        case 0x0B: // MADDAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * bc;
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x0C:
        case 0x0D:
        case 0x0E:
        case 0x0F: // MSUBAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * bc;
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x10: // ITOF0
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x11: // ITOF4
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 16.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x12: // ITOF12
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 4096.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x13: // ITOF15
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 32768.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x14: // FTOI0
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 1.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x15: // FTOI4
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 16.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x16: // FTOI12
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 4096.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x17: // FTOI15
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 32768.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x18:
        case 0x19:
        case 0x1A:
        case 0x1B: // MULAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * bc;
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x1C: // MULAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * q;
            applyFmacDestAcc(result, dest);
            return;
        case 0x1D: // ABS (clears the sign bit; the magnitude bits pass through)
            for (int c = 0; c < 4; c++)
            {
                uint32_t bits = 0u;
                std::memcpy(&bits, &m_state.vf[fs][c], sizeof(bits));
                bits &= 0x7FFFFFFFu;
                std::memcpy(&result[c], &bits, sizeof(bits));
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x1E: // MULAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * i;
            applyFmacDestAcc(result, dest);
            return;
        case 0x1F: // CLIP
        {
            uint32_t wBits = 0u;
            std::memcpy(&wBits, &m_state.vf[ft][3], sizeof(wBits));
            const int32_t limit = (wBits & 0x7F800000u) != 0u ? static_cast<int32_t>(wBits & 0x7FFFFFFFu) : 0x007FFFFF;

            const auto exceedsClipPlane = [limit](float value, uint32_t signMask)
            {
                uint32_t bits = 0u;
                std::memcpy(&bits, &value, sizeof(bits));
                bits ^= signMask;
                int32_t orderedBits = 0;
                std::memcpy(&orderedBits, &bits, sizeof(orderedBits));
                return orderedBits > limit;
            };

            uint32_t flags = 0u;
            if (exceedsClipPlane(m_state.vf[fs][0], 0x00000000u))
                flags |= 0x01u;
            if (exceedsClipPlane(m_state.vf[fs][0], 0x80000000u))
                flags |= 0x02u;
            if (exceedsClipPlane(m_state.vf[fs][1], 0x00000000u))
                flags |= 0x04u;
            if (exceedsClipPlane(m_state.vf[fs][1], 0x80000000u))
                flags |= 0x08u;
            if (exceedsClipPlane(m_state.vf[fs][2], 0x00000000u))
                flags |= 0x10u;
            if (exceedsClipPlane(m_state.vf[fs][2], 0x80000000u))
                flags |= 0x20u;
            queueClip(flags);
            return;
        }
        case 0x20: // ADDAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + q;
            applyFmacDestAcc(result, dest);
            return;
        case 0x21: // MADDAq
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * q;
            applyFmacDestAcc(result, dest);
            return;
        case 0x22: // ADDAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + i;
            applyFmacDestAcc(result, dest);
            return;
        case 0x23: // MADDAi
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * i;
            applyFmacDestAcc(result, dest);
            return;
        case 0x24: // SUBAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - q;
            applyFmacDestAcc(result, dest);
            return;
        case 0x25: // MSUBAq
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * q;
            applyFmacDestAcc(result, dest);
            return;
        case 0x26: // SUBAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - i;
            applyFmacDestAcc(result, dest);
            return;
        case 0x27: // MSUBAi
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * i;
            applyFmacDestAcc(result, dest);
            return;
        case 0x28: // ADDA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x29: // MADDA
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x2A: // MULA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x2C: // SUBA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x2D: // MSUBA
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x2E: // OPMULA
            result[0] = vs[1] * vt[2];
            result[1] = vs[2] * vt[0];
            result[2] = vs[0] * vt[1];
            result[3] = 0.0f;
            applyFmacDestAcc(result, dest);
            return;
        case 0x2F:
        case 0x30: // NOP
            return;
        default:
            reportReservedInstruction(true, instr);
            return;
        }
    }

    case 0x30:
    case 0x31:
    case 0x32:
    case 0x33:
    default:
        reportReservedInstruction(true, instr);
        return;
    }
}
