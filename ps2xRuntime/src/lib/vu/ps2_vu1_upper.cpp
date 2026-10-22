#include "runtime/ps2_vu1.h"
#include "ps2_vu1_detail.h"

#include <cmath>
#include <emmintrin.h>
#include <immintrin.h>
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

    // PS2 FMAC addition. Before adding, the smaller operand loses the mantissa bits below the
    // larger one's last bit (no guard bits); the sum then rounds toward zero, which the
    // interpreter's FE_TOWARDZERO mode does. A plain IEEE sum differs by one ulp when the small
    // operand has the opposite sign: Valkyrie Profile 2 derives its file decryption key on VU0
    // and depends on the exact PS2 result (PCSX2 hardcodes these additions as a "TriAce hack").
    float vuAdd(float a, float b)
    {

        uint32_t ua = 0u;
        uint32_t ub = 0u;
        std::memcpy(&ua, &a, sizeof(ua));
        std::memcpy(&ub, &b, sizeof(ub));
        const uint32_t ea = (ua >> 23) & 0xFFu;
        const uint32_t eb = (ub >> 23) & 0xFFu;
        if (ea != 0u && eb != 0u && ea != eb)
        {
            uint32_t &small = ea > eb ? ub : ua;
            const uint32_t shift = ea > eb ? ea - eb : eb - ea;
            small &= shift >= 24u ? 0x80000000u : ~((1u << shift) - 1u);
            std::memcpy(&a, &ua, sizeof(a));
            std::memcpy(&b, &ub, sizeof(b));
        }
        return a + b;
    }

    float vuSub(float a, float b)
    {
        return vuAdd(a, -b);
    }

    // vuAdd on four lanes: with AVX2 the same operand alignment, then one SSE add under the
    // interpreter's round-toward-zero mode; without it, lane by lane.
    __m128 vuAdd4(__m128 a, __m128 b)
    {
#if !defined(__AVX2__)
        alignas(16) float la[4];
        alignas(16) float lb[4];
        _mm_store_ps(la, a);
        _mm_store_ps(lb, b);
        for (int c = 0; c < 4; ++c)
            la[c] = vuAdd(la[c], lb[c]);
        return _mm_load_ps(la);
#else
        const __m128i ia = _mm_castps_si128(a);
        const __m128i ib = _mm_castps_si128(b);
        const __m128i exponentMask = _mm_set1_epi32(0xFF);
        const __m128i allOnes = _mm_set1_epi32(-1);
        const __m128i ea = _mm_and_si128(_mm_srli_epi32(ia, 23), exponentMask);
        const __m128i eb = _mm_and_si128(_mm_srli_epi32(ib, 23), exponentMask);
        const __m128i zero = _mm_setzero_si128();
        const __m128i inactive = _mm_or_si128(_mm_or_si128(_mm_cmpeq_epi32(ea, zero), _mm_cmpeq_epi32(eb, zero)),
                                              _mm_cmpeq_epi32(ea, eb));
        const __m128i aLarger = _mm_cmpgt_epi32(ea, eb);
        const __m128i shift = _mm_abs_epi32(_mm_sub_epi32(ea, eb));
        const __m128i one = _mm_set1_epi32(1);
        const __m128i lowBits = _mm_sub_epi32(_mm_sllv_epi32(one, shift), one);
        const __m128i far = _mm_cmpgt_epi32(shift, _mm_set1_epi32(23));
        const __m128i keep = _mm_or_si128(_mm_andnot_si128(far, _mm_xor_si128(lowBits, allOnes)),
                                          _mm_and_si128(far, _mm_set1_epi32(static_cast<int>(0x80000000u))));
        // The operand with the smaller exponent loses its low bits.
        const __m128i truncateA = _mm_andnot_si128(inactive, _mm_andnot_si128(aLarger, allOnes));
        const __m128i truncateB = _mm_andnot_si128(inactive, aLarger);
        const __m128i newA = _mm_and_si128(ia, _mm_or_si128(_mm_andnot_si128(truncateA, allOnes), keep));
        const __m128i newB = _mm_and_si128(ib, _mm_or_si128(_mm_andnot_si128(truncateB, allOnes), keep));
        return _mm_add_ps(_mm_castsi128_ps(newA), _mm_castsi128_ps(newB));
#endif
    }

    __m128 negate4(__m128 value)
    {
        return _mm_xor_ps(value, _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>(0x80000000u))));
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
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), _mm_set1_ps(bc)));
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x04:
    case 0x05:
    case 0x06:
    case 0x07: // SUBbc
    {
        float bc = broadcast(vt, op & 3);
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), negate4(_mm_set1_ps(bc))));
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x08:
    case 0x09:
    case 0x0A:
    case 0x0B: // MADDbc
    {
        float bc = broadcast(vt, op & 3);
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), _mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(bc))));
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x0C:
    case 0x0D:
    case 0x0E:
    case 0x0F: // MSUBbc
    {
        float bc = broadcast(vt, op & 3);
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), negate4(_mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(bc)))));
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
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), _mm_set1_ps(q)));
        applyFmacDest(vd, result, dest);
        return;
    case 0x21: // MADDq
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), _mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(q))));
        applyFmacDest(vd, result, dest);
        return;
    case 0x22: // ADDi
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), _mm_set1_ps(i)));
        applyFmacDest(vd, result, dest);
        return;
    case 0x23: // MADDi
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), _mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(i))));
        applyFmacDest(vd, result, dest);
        return;
    case 0x24: // SUBq
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), negate4(_mm_set1_ps(q))));
        applyFmacDest(vd, result, dest);
        return;
    case 0x25: // MSUBq
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), negate4(_mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(q)))));
        applyFmacDest(vd, result, dest);
        return;
    case 0x26: // SUBi
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), negate4(_mm_set1_ps(i))));
        applyFmacDest(vd, result, dest);
        return;
    case 0x27: // MSUBi
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), negate4(_mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(i)))));
        applyFmacDest(vd, result, dest);
        return;
    case 0x28: // ADD
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), _mm_load_ps(vt)));
        applyFmacDest(vd, result, dest);
        return;
    case 0x29: // MADD
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), _mm_mul_ps(_mm_load_ps(vs), _mm_load_ps(vt))));
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
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), negate4(_mm_load_ps(vt))));
        applyFmacDest(vd, result, dest);
        return;
    case 0x2D: // MSUB
        _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), negate4(_mm_mul_ps(_mm_load_ps(vs), _mm_load_ps(vt)))));
        applyFmacDest(vd, result, dest);
        return;
    case 0x2E: // OPMSUB
        result[0] = vuSub(acc[0], vs[1] * vt[2]);
        result[1] = vuSub(acc[1], vs[2] * vt[0]);
        result[2] = vuSub(acc[2], vs[0] * vt[1]);
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
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), _mm_set1_ps(bc)));
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x04:
        case 0x05:
        case 0x06:
        case 0x07: // SUBAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), negate4(_mm_set1_ps(bc))));
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x08:
        case 0x09:
        case 0x0A:
        case 0x0B: // MADDAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), _mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(bc))));
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x0C:
        case 0x0D:
        case 0x0E:
        case 0x0F: // MSUBAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), negate4(_mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(bc)))));
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
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), _mm_set1_ps(q)));
            applyFmacDestAcc(result, dest);
            return;
        case 0x21: // MADDAq
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), _mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(q))));
            applyFmacDestAcc(result, dest);
            return;
        case 0x22: // ADDAi
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), _mm_set1_ps(i)));
            applyFmacDestAcc(result, dest);
            return;
        case 0x23: // MADDAi
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), _mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(i))));
            applyFmacDestAcc(result, dest);
            return;
        case 0x24: // SUBAq
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), negate4(_mm_set1_ps(q))));
            applyFmacDestAcc(result, dest);
            return;
        case 0x25: // MSUBAq
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), negate4(_mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(q)))));
            applyFmacDestAcc(result, dest);
            return;
        case 0x26: // SUBAi
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), negate4(_mm_set1_ps(i))));
            applyFmacDestAcc(result, dest);
            return;
        case 0x27: // MSUBAi
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), negate4(_mm_mul_ps(_mm_load_ps(vs), _mm_set1_ps(i)))));
            applyFmacDestAcc(result, dest);
            return;
        case 0x28: // ADDA
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), _mm_load_ps(vt)));
            applyFmacDestAcc(result, dest);
            return;
        case 0x29: // MADDA
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), _mm_mul_ps(_mm_load_ps(vs), _mm_load_ps(vt))));
            applyFmacDestAcc(result, dest);
            return;
        case 0x2A: // MULA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x2C: // SUBA
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(vs), negate4(_mm_load_ps(vt))));
            applyFmacDestAcc(result, dest);
            return;
        case 0x2D: // MSUBA
            _mm_storeu_ps(result, vuAdd4(_mm_load_ps(acc), negate4(_mm_mul_ps(_mm_load_ps(vs), _mm_load_ps(vt)))));
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
