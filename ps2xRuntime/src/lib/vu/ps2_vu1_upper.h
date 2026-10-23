#ifndef PS2_VU1_UPPER_H
#define PS2_VU1_UPPER_H

#include "runtime/ps2_vu1.h"
#include "ps2_vu1_detail.h"

#include <array>
#include <cmath>
#include <emmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <cstring>
#include <limits>

namespace ps2_vu1_upper_ops
{
    inline int32_t vuFloatToInt(float value, float scale)
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
    inline float vuAdd(float a, float b)
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

    inline float vuSub(float a, float b)
    {
        return vuAdd(a, -b);
    }

    // MAX/MINI compare raw sign-magnitude bit patterns and pass the chosen operand through unchanged, so
    // denormals (often integer data, e.g. colors unpacked by VIF) and exponent-255 values survive.
    inline int64_t vuMinMaxKey(float value)
    {
        uint32_t bits = 0u;
        std::memcpy(&bits, &value, sizeof(bits));
        const int64_t magnitude = static_cast<int64_t>(bits & 0x7FFFFFFFu);
        return (bits & 0x80000000u) != 0u ? -magnitude : magnitude;
    }

    inline float vuMax(float a, float b) { return vuMinMaxKey(a) > vuMinMaxKey(b) ? a : b; }
    inline float vuMini(float a, float b) { return vuMinMaxKey(a) < vuMinMaxKey(b) ? a : b; }

    // normalizeOperand() on four lanes: denormals become signed zero, exponent 255 becomes +-FLT_MAX.
    inline void normalizeVector(const float *source, float *target)
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

namespace ps2_vu1_upper_ops
{
    // normalizeVector() on a register.
    inline __m128 normalizeLanes(__m128 value)
    {
        const __m128i bits = _mm_castps_si128(value);
        const __m128i exponentMask = _mm_set1_epi32(0x7F800000);
        const __m128i sign = _mm_and_si128(bits, _mm_set1_epi32(static_cast<int>(0x80000000u)));
        const __m128i exponent = _mm_and_si128(bits, exponentMask);
        const __m128i zeroExponent = _mm_cmpeq_epi32(exponent, _mm_setzero_si128());
        const __m128i maxExponent = _mm_cmpeq_epi32(exponent, exponentMask);
        __m128i result = _mm_or_si128(_mm_andnot_si128(zeroExponent, bits), _mm_and_si128(zeroExponent, sign));
        result = _mm_or_si128(_mm_andnot_si128(maxExponent, result),
                              _mm_and_si128(maxExponent, _mm_or_si128(sign, _mm_set1_epi32(0x7F7FFFFF))));
        return _mm_castsi128_ps(result);
    }

    // vuAdd on four lanes. 1 << shift comes from converting 2^shift, since SSE4 has no variable shift.
    __attribute__((always_inline)) inline __m128 vuAdd4(__m128 a, __m128 b)
    {
        const __m128i ua = _mm_castps_si128(a);
        const __m128i ub = _mm_castps_si128(b);
        const __m128i ea = _mm_srli_epi32(_mm_slli_epi32(ua, 1), 24);
        const __m128i eb = _mm_srli_epi32(_mm_slli_epi32(ub, 1), 24);
        const __m128i zero = _mm_setzero_si128();
        const __m128i ones = _mm_set1_epi32(-1);
        const __m128i bothNonZero = _mm_andnot_si128(_mm_or_si128(_mm_cmpeq_epi32(ea, zero), _mm_cmpeq_epi32(eb, zero)), ones);
        const __m128i shift = _mm_abs_epi32(_mm_sub_epi32(ea, eb));
        const __m128i power = _mm_cvttps_epi32(
            _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(_mm_min_epi32(shift, _mm_set1_epi32(24)), _mm_set1_epi32(127)), 23)));
        const __m128i wide = _mm_cmpgt_epi32(shift, _mm_set1_epi32(23));
        const __m128i smallMask = _mm_or_si128(_mm_andnot_si128(wide, _mm_xor_si128(_mm_sub_epi32(power, _mm_set1_epi32(1)), ones)),
                                               _mm_and_si128(wide, _mm_set1_epi32(static_cast<int>(0x80000000u))));
        const __m128i aSmall = _mm_and_si128(bothNonZero, _mm_cmpgt_epi32(eb, ea));
        const __m128i bSmall = _mm_and_si128(bothNonZero, _mm_cmpgt_epi32(ea, eb));
        const __m128i maskedA = _mm_and_si128(ua, _mm_or_si128(_mm_andnot_si128(aSmall, ones), _mm_and_si128(aSmall, smallMask)));
        const __m128i maskedB = _mm_and_si128(ub, _mm_or_si128(_mm_andnot_si128(bSmall, ones), _mm_and_si128(bSmall, smallMask)));
        return _mm_add_ps(_mm_castsi128_ps(maskedA), _mm_castsi128_ps(maskedB));
    }

    inline __m128 destLaneMask(uint8_t dest)
    {
        return _mm_castsi128_ps(_mm_set_epi32((dest & 1u) ? -1 : 0, (dest & 2u) ? -1 : 0, (dest & 4u) ? -1 : 0,
                                              (dest & 8u) ? -1 : 0));
    }
}
using namespace ps2_vu1_upper_ops;

inline void VU1Interpreter::execUpperNop(uint32_t instr)
{
    m_currentUpperInstruction = instr;
}

template <VU1Interpreter::FmacKind Kind, VU1Interpreter::FmacOperand Operand, bool ToAcc>
inline void VU1Interpreter::execFmac(uint32_t instr)
{
    m_currentUpperInstruction = instr;
    const uint8_t dest = DEST(instr);
    alignas(16) float vs[4];
    alignas(16) float vt[4];
    alignas(16) float acc[4];
    float scalar = 0.0f;
    const __m128 s = normalizeLanes(_mm_loadu_ps(m_state.vf[FS(instr)]));
    _mm_store_ps(vs, s);
    __m128 t;
    if constexpr (Operand == FmacOperand::Vector || Operand == FmacOperand::Broadcast)
    {
        _mm_store_ps(vt, normalizeLanes(_mm_loadu_ps(m_state.vf[FT(instr)])));
        t = Operand == FmacOperand::Vector ? _mm_load_ps(vt) : _mm_set1_ps(vt[instr & 3u]);
    }
    else
    {
        scalar = normalizeOperand(Operand == FmacOperand::Q ? m_state.q : m_state.i);
        t = _mm_set1_ps(scalar);
    }

    const __m128 signBit = _mm_castsi128_ps(_mm_set1_epi32(static_cast<int>(0x80000000u)));
    __m128 r;
    if constexpr (Kind == FmacKind::Mul)
        r = _mm_mul_ps(s, t);
    else if constexpr (Kind == FmacKind::Add)
        r = vuAdd4(s, t);
    else if constexpr (Kind == FmacKind::Sub)
        r = vuAdd4(s, _mm_xor_ps(t, signBit));
    else
    {
        const __m128 a = normalizeLanes(_mm_loadu_ps(m_state.acc));
        _mm_store_ps(acc, a);
        const __m128 product = _mm_mul_ps(s, t);
        r = vuAdd4(a, Kind == FmacKind::Madd ? product : _mm_xor_ps(product, signBit));
    }

    alignas(16) float result[4];
    _mm_store_ps(result, r);
    bool exact = m_currentFlagsLive;
    if (!exact)
    {
        const __m128i exponent = _mm_srli_epi32(_mm_slli_epi32(_mm_castps_si128(r), 1), 24);
        const __m128i ordinary = _mm_and_si128(_mm_cmpgt_epi32(exponent, _mm_set1_epi32(1)),
                                               _mm_cmplt_epi32(exponent, _mm_set1_epi32(254)));
        const int lanes = _mm_movemask_ps(destLaneMask(dest));
        // An exact zero (see exactZeroLanes): any sum or difference, or a product with a zero factor.
        int zero = _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(_mm_slli_epi32(_mm_castps_si128(r), 1), _mm_setzero_si128())));
        if constexpr (Kind == FmacKind::Mul || Kind == FmacKind::Madd || Kind == FmacKind::Msub)
            zero &= _mm_movemask_ps(_mm_or_ps(_mm_cmpeq_ps(s, _mm_setzero_ps()), _mm_cmpeq_ps(t, _mm_setzero_ps())));
        exact = ((_mm_movemask_ps(_mm_castsi128_ps(ordinary)) | zero) & lanes) != lanes;
    }
    if (exact)
    {
        if constexpr (Operand == FmacOperand::Q || Operand == FmacOperand::I)
            normalizeVector(m_state.vf[FT(instr)], vt);
        m_fmacVs = vs;
        m_fmacVt = vt;
        m_fmacAcc = acc;
        m_fmacQ = Operand == FmacOperand::Q ? scalar : 0.0f;
        m_fmacI = Operand == FmacOperand::I ? scalar : 0.0f;
        applyFmacFlags(result, dest);
        r = _mm_load_ps(result);
    }
    float *target = ToAcc ? m_state.acc : m_state.vf[FD(instr)];
    _mm_storeu_ps(target, _mm_blendv_ps(_mm_loadu_ps(target), r, destLaneMask(dest)));
}

inline VU1Interpreter::UpperHandler VU1Interpreter::selectUpperHandler(uint32_t upper)
{
    using K = FmacKind;
    using O = FmacOperand;
    const uint32_t low11 = upper & 0x7FFu;
    if (low11 == 0x2FFu || low11 == 0x33Cu)
        return &VU1Interpreter::execUpperNop;
    const uint32_t op = upper & 0x3Fu;
    const bool special = op >= 0x3Cu;
    const uint32_t code = special ? ((upper & 3u) | ((upper >> 4) & 0x7Cu)) : op;
    const auto pick = [special](UpperHandler toVf, UpperHandler toAcc) { return special ? toAcc : toVf; };
#define PS2X_FMAC(kind, operand) pick(&VU1Interpreter::execFmac<K::kind, O::operand, false>, &VU1Interpreter::execFmac<K::kind, O::operand, true>)
    if (code <= 0x03u)
        return PS2X_FMAC(Add, Broadcast);
    if (code <= 0x07u)
        return PS2X_FMAC(Sub, Broadcast);
    if (code <= 0x0Bu)
        return PS2X_FMAC(Madd, Broadcast);
    if (code <= 0x0Fu)
        return PS2X_FMAC(Msub, Broadcast);
    switch (code)
    {
    case 0x18u:
    case 0x19u:
    case 0x1Au:
    case 0x1Bu:
        return PS2X_FMAC(Mul, Broadcast);
    case 0x1Cu:
        return PS2X_FMAC(Mul, Q);
    case 0x1Eu:
        return PS2X_FMAC(Mul, I);
    case 0x20u:
        return PS2X_FMAC(Add, Q);
    case 0x21u:
        return PS2X_FMAC(Madd, Q);
    case 0x22u:
        return PS2X_FMAC(Add, I);
    case 0x23u:
        return PS2X_FMAC(Madd, I);
    case 0x24u:
        return PS2X_FMAC(Sub, Q);
    case 0x25u:
        return PS2X_FMAC(Msub, Q);
    case 0x26u:
        return PS2X_FMAC(Sub, I);
    case 0x27u:
        return PS2X_FMAC(Msub, I);
    case 0x28u:
        return PS2X_FMAC(Add, Vector);
    case 0x29u:
        return PS2X_FMAC(Madd, Vector);
    case 0x2Au:
        return PS2X_FMAC(Mul, Vector);
    case 0x2Cu:
        return PS2X_FMAC(Sub, Vector);
    case 0x2Du:
        return PS2X_FMAC(Msub, Vector);
    default:
        return &VU1Interpreter::execUpper;
    }
#undef PS2X_FMAC
}

// ============================================================================
// Upper instructions (FMAC pipeline)
// ============================================================================
inline void VU1Interpreter::execUpper(uint32_t instr)
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

    // Which of ACC, Q and I an upper op reads, by op (0x00-0x3B) or 0x40 | special op.
    static constexpr auto kReads = []
    {
        constexpr uint8_t kAcc = 1u, kQ = 2u, kI = 4u;
        std::array<uint8_t, 0x80> reads{};
        for (const uint32_t base : {0x00u, 0x40u})
        {
            for (uint32_t code = 0x08u; code <= 0x0Fu; ++code)
                reads[base | code] |= kAcc;
            for (const uint32_t code : {0x21u, 0x23u, 0x25u, 0x27u, 0x29u, 0x2Du})
                reads[base | code] |= kAcc;
            for (const uint32_t code : {0x1Cu, 0x20u, 0x21u, 0x24u, 0x25u})
                reads[base | code] |= kQ;
            for (const uint32_t code : {0x1Eu, 0x22u, 0x23u, 0x26u, 0x27u})
                reads[base | code] |= kI;
        }
        reads[0x2Eu] |= kAcc; // OPMSUB
        return reads;
    }();
    const uint8_t reads =
        kReads[op >= 0x3Cu ? (0x40u | ((instr & 3u) | ((instr >> 4) & 0x7Cu))) & 0x7Fu : op];

    float *vd = m_state.vf[fd];
    alignas(16) float normalizedVs[4];
    alignas(16) float normalizedVt[4];
    alignas(16) float normalizedAcc[4];
    normalizeVector(m_state.vf[fs], normalizedVs);
    normalizeVector(m_state.vf[ft], normalizedVt);
    if (reads & 1u)
        normalizeVector(m_state.acc, normalizedAcc);
    const float *vs = normalizedVs;
    const float *vt = normalizedVt;
    const float *acc = normalizedAcc;
    const float q = (reads & 2u) ? normalizeOperand(m_state.q) : 0.0f;
    const float i = (reads & 4u) ? normalizeOperand(m_state.i) : 0.0f;
    float result[4];
    m_fmacVs = vs;
    m_fmacVt = vt;
    m_fmacAcc = acc;
    m_fmacQ = q;
    m_fmacI = i;

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
            result[c] = vuAdd(vs[c], bc);
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
            result[c] = vuSub(vs[c], bc);
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
            result[c] = vuAdd(acc[c], vs[c] * bc);
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
            result[c] = vuSub(acc[c], vs[c] * bc);
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
            result[c] = vuAdd(vs[c], q);
        applyFmacDest(vd, result, dest);
        return;
    case 0x21: // MADDq
        for (int c = 0; c < 4; c++)
            result[c] = vuAdd(acc[c], vs[c] * q);
        applyFmacDest(vd, result, dest);
        return;
    case 0x22: // ADDi
        for (int c = 0; c < 4; c++)
            result[c] = vuAdd(vs[c], i);
        applyFmacDest(vd, result, dest);
        return;
    case 0x23: // MADDi
        for (int c = 0; c < 4; c++)
            result[c] = vuAdd(acc[c], vs[c] * i);
        applyFmacDest(vd, result, dest);
        return;
    case 0x24: // SUBq
        for (int c = 0; c < 4; c++)
            result[c] = vuSub(vs[c], q);
        applyFmacDest(vd, result, dest);
        return;
    case 0x25: // MSUBq
        for (int c = 0; c < 4; c++)
            result[c] = vuSub(acc[c], vs[c] * q);
        applyFmacDest(vd, result, dest);
        return;
    case 0x26: // SUBi
        for (int c = 0; c < 4; c++)
            result[c] = vuSub(vs[c], i);
        applyFmacDest(vd, result, dest);
        return;
    case 0x27: // MSUBi
        for (int c = 0; c < 4; c++)
            result[c] = vuSub(acc[c], vs[c] * i);
        applyFmacDest(vd, result, dest);
        return;
    case 0x28: // ADD
        for (int c = 0; c < 4; c++)
            result[c] = vuAdd(vs[c], vt[c]);
        applyFmacDest(vd, result, dest);
        return;
    case 0x29: // MADD
        for (int c = 0; c < 4; c++)
            result[c] = vuAdd(acc[c], vs[c] * vt[c]);
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
            result[c] = vuSub(vs[c], vt[c]);
        applyFmacDest(vd, result, dest);
        return;
    case 0x2D: // MSUB
        for (int c = 0; c < 4; c++)
            result[c] = vuSub(acc[c], vs[c] * vt[c]);
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
            for (int c = 0; c < 4; c++)
                result[c] = vuAdd(vs[c], bc);
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
                result[c] = vuSub(vs[c], bc);
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
                result[c] = vuAdd(acc[c], vs[c] * bc);
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
                result[c] = vuSub(acc[c], vs[c] * bc);
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
                result[c] = vuAdd(vs[c], q);
            applyFmacDestAcc(result, dest);
            return;
        case 0x21: // MADDAq
            for (int c = 0; c < 4; c++)
                result[c] = vuAdd(acc[c], vs[c] * q);
            applyFmacDestAcc(result, dest);
            return;
        case 0x22: // ADDAi
            for (int c = 0; c < 4; c++)
                result[c] = vuAdd(vs[c], i);
            applyFmacDestAcc(result, dest);
            return;
        case 0x23: // MADDAi
            for (int c = 0; c < 4; c++)
                result[c] = vuAdd(acc[c], vs[c] * i);
            applyFmacDestAcc(result, dest);
            return;
        case 0x24: // SUBAq
            for (int c = 0; c < 4; c++)
                result[c] = vuSub(vs[c], q);
            applyFmacDestAcc(result, dest);
            return;
        case 0x25: // MSUBAq
            for (int c = 0; c < 4; c++)
                result[c] = vuSub(acc[c], vs[c] * q);
            applyFmacDestAcc(result, dest);
            return;
        case 0x26: // SUBAi
            for (int c = 0; c < 4; c++)
                result[c] = vuSub(vs[c], i);
            applyFmacDestAcc(result, dest);
            return;
        case 0x27: // MSUBAi
            for (int c = 0; c < 4; c++)
                result[c] = vuSub(acc[c], vs[c] * i);
            applyFmacDestAcc(result, dest);
            return;
        case 0x28: // ADDA
            for (int c = 0; c < 4; c++)
                result[c] = vuAdd(vs[c], vt[c]);
            applyFmacDestAcc(result, dest);
            return;
        case 0x29: // MADDA
            for (int c = 0; c < 4; c++)
                result[c] = vuAdd(acc[c], vs[c] * vt[c]);
            applyFmacDestAcc(result, dest);
            return;
        case 0x2A: // MULA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x2C: // SUBA
            for (int c = 0; c < 4; c++)
                result[c] = vuSub(vs[c], vt[c]);
            applyFmacDestAcc(result, dest);
            return;
        case 0x2D: // MSUBA
            for (int c = 0; c < 4; c++)
                result[c] = vuSub(acc[c], vs[c] * vt[c]);
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

#endif
