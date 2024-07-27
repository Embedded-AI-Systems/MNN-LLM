//
//  Vec16.hpp
//  MNN
//
//  Created by MNN on 2021/05/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Vec16_hpp
#define Vec16_hpp
#include "FunctionSummary.hpp"


inline void transpose16x16(
                   __m512i& r0, __m512i& r1, __m512i& r2, __m512i& r3,
                   __m512i& r4, __m512i& r5, __m512i& r6, __m512i& r7,
                   __m512i& r8, __m512i& r9, __m512i& ra, __m512i& rb,
                   __m512i& rc, __m512i& rd, __m512i& re, __m512i& rf) {
    //given __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;

    t0 = _mm512_unpacklo_epi32(r0,r1); //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29
    t1 = _mm512_unpackhi_epi32(r0,r1); //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
    t2 = _mm512_unpacklo_epi32(r2,r3); //  32  48  33  49 ...
    t3 = _mm512_unpackhi_epi32(r2,r3); //  34  50  35  51 ...
    t4 = _mm512_unpacklo_epi32(r4,r5); //  64  80  65  81 ...
    t5 = _mm512_unpackhi_epi32(r4,r5); //  66  82  67  83 ...
    t6 = _mm512_unpacklo_epi32(r6,r7); //  96 112  97 113 ...
    t7 = _mm512_unpackhi_epi32(r6,r7); //  98 114  99 115 ...
    t8 = _mm512_unpacklo_epi32(r8,r9); // 128 ...
    t9 = _mm512_unpackhi_epi32(r8,r9); // 130 ...
    ta = _mm512_unpacklo_epi32(ra,rb); // 160 ...
    tb = _mm512_unpackhi_epi32(ra,rb); // 162 ...
    tc = _mm512_unpacklo_epi32(rc,rd); // 196 ...
    td = _mm512_unpackhi_epi32(rc,rd); // 198 ...
    te = _mm512_unpacklo_epi32(re,rf); // 228 ...
    tf = _mm512_unpackhi_epi32(re,rf); // 230 ...

    r0 = _mm512_unpacklo_epi64(t0,t2); //   0  16  32  48 ...
    r1 = _mm512_unpackhi_epi64(t0,t2); //   1  17  33  49 ...
    r2 = _mm512_unpacklo_epi64(t1,t3); //   2  18  34  49 ...
    r3 = _mm512_unpackhi_epi64(t1,t3); //   3  19  35  51 ...
    r4 = _mm512_unpacklo_epi64(t4,t6); //  64  80  96 112 ...
    r5 = _mm512_unpackhi_epi64(t4,t6); //  65  81  97 114 ...
    r6 = _mm512_unpacklo_epi64(t5,t7); //  66  82  98 113 ...
    r7 = _mm512_unpackhi_epi64(t5,t7); //  67  83  99 115 ...
    r8 = _mm512_unpacklo_epi64(t8,ta); // 128 144 160 176 ...
    r9 = _mm512_unpackhi_epi64(t8,ta); // 129 145 161 178 ...
    ra = _mm512_unpacklo_epi64(t9,tb); // 130 146 162 177 ...
    rb = _mm512_unpackhi_epi64(t9,tb); // 131 147 163 179 ...
    rc = _mm512_unpacklo_epi64(tc,te); // 192 208 228 240 ...
    rd = _mm512_unpackhi_epi64(tc,te); // 193 209 229 241 ...
    re = _mm512_unpacklo_epi64(td,tf); // 194 210 230 242 ...
    rf = _mm512_unpackhi_epi64(td,tf); // 195 211 231 243 ...

    t0 = _mm512_shuffle_i32x4(r0, r4, 0x88); //   0  16  32  48   8  24  40  56  64  80  96  112 ...
    t1 = _mm512_shuffle_i32x4(r1, r5, 0x88); //   1  17  33  49 ...
    t2 = _mm512_shuffle_i32x4(r2, r6, 0x88); //   2  18  34  50 ...
    t3 = _mm512_shuffle_i32x4(r3, r7, 0x88); //   3  19  35  51 ...
    t4 = _mm512_shuffle_i32x4(r0, r4, 0xdd); //   4  20  36  52 ...
    t5 = _mm512_shuffle_i32x4(r1, r5, 0xdd); //   5  21  37  53 ...
    t6 = _mm512_shuffle_i32x4(r2, r6, 0xdd); //   6  22  38  54 ...
    t7 = _mm512_shuffle_i32x4(r3, r7, 0xdd); //   7  23  39  55 ...
    t8 = _mm512_shuffle_i32x4(r8, rc, 0x88); // 128 144 160 176 ...
    t9 = _mm512_shuffle_i32x4(r9, rd, 0x88); // 129 145 161 177 ...
    ta = _mm512_shuffle_i32x4(ra, re, 0x88); // 130 146 162 178 ...
    tb = _mm512_shuffle_i32x4(rb, rf, 0x88); // 131 147 163 179 ...
    tc = _mm512_shuffle_i32x4(r8, rc, 0xdd); // 132 148 164 180 ...
    td = _mm512_shuffle_i32x4(r9, rd, 0xdd); // 133 149 165 181 ...
    te = _mm512_shuffle_i32x4(ra, re, 0xdd); // 134 150 166 182 ...
    tf = _mm512_shuffle_i32x4(rb, rf, 0xdd); // 135 151 167 183 ...

    r0 = _mm512_shuffle_i32x4(t0, t8, 0x88); //   0  16  32  48  64  80  96 112 ... 240
    r1 = _mm512_shuffle_i32x4(t1, t9, 0x88); //   1  17  33  49  66  81  97 113 ... 241
    r2 = _mm512_shuffle_i32x4(t2, ta, 0x88); //   2  18  34  50  67  82  98 114 ... 242
    r3 = _mm512_shuffle_i32x4(t3, tb, 0x88); //   3  19  35  51  68  83  99 115 ... 243
    r4 = _mm512_shuffle_i32x4(t4, tc, 0x88); //   4 ...
    r5 = _mm512_shuffle_i32x4(t5, td, 0x88); //   5 ...
    r6 = _mm512_shuffle_i32x4(t6, te, 0x88); //   6 ...
    r7 = _mm512_shuffle_i32x4(t7, tf, 0x88); //   7 ...
    r8 = _mm512_shuffle_i32x4(t0, t8, 0xdd); //   8 ...
    r9 = _mm512_shuffle_i32x4(t1, t9, 0xdd); //   9 ...
    ra = _mm512_shuffle_i32x4(t2, ta, 0xdd); //  10 ...
    rb = _mm512_shuffle_i32x4(t3, tb, 0xdd); //  11 ...
    rc = _mm512_shuffle_i32x4(t4, tc, 0xdd); //  12 ...
    rd = _mm512_shuffle_i32x4(t5, td, 0xdd); //  13 ...
    re = _mm512_shuffle_i32x4(t6, te, 0xdd); //  14 ...
    rf = _mm512_shuffle_i32x4(t7, tf, 0xdd); //  15  31  47  63  79  96 111 127 ... 255
}

inline void transpose16x16F(
                   __m512& r0f, __m512& r1f, __m512& r2f, __m512& r3f,
                   __m512& r4f, __m512& r5f, __m512& r6f, __m512& r7f,
                   __m512& r8f, __m512& r9f, __m512& raf, __m512& rbf,
                   __m512& rcf, __m512& rdf, __m512& ref, __m512& rff) {
    auto r0 = _mm512_castps_si512(r0f);
    auto r1 = _mm512_castps_si512(r1f);
    auto r2 = _mm512_castps_si512(r2f);
    auto r3 = _mm512_castps_si512(r3f);
    auto r4 = _mm512_castps_si512(r4f);
    auto r5 = _mm512_castps_si512(r5f);
    auto r6 = _mm512_castps_si512(r6f);
    auto r7 = _mm512_castps_si512(r7f);
    auto r8 = _mm512_castps_si512(r8f);
    auto r9 = _mm512_castps_si512(r9f);
    auto ra = _mm512_castps_si512(raf);
    auto rb = _mm512_castps_si512(rbf);
    auto rc = _mm512_castps_si512(rcf);
    auto rd = _mm512_castps_si512(rdf);
    auto re = _mm512_castps_si512(ref);
    auto rf = _mm512_castps_si512(rff);
    //given __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;

    t0 = _mm512_unpacklo_epi32(r0,r1); //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29
    t1 = _mm512_unpackhi_epi32(r0,r1); //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
    t2 = _mm512_unpacklo_epi32(r2,r3); //  32  48  33  49 ...
    t3 = _mm512_unpackhi_epi32(r2,r3); //  34  50  35  51 ...
    t4 = _mm512_unpacklo_epi32(r4,r5); //  64  80  65  81 ...
    t5 = _mm512_unpackhi_epi32(r4,r5); //  66  82  67  83 ...
    t6 = _mm512_unpacklo_epi32(r6,r7); //  96 112  97 113 ...
    t7 = _mm512_unpackhi_epi32(r6,r7); //  98 114  99 115 ...
    t8 = _mm512_unpacklo_epi32(r8,r9); // 128 ...
    t9 = _mm512_unpackhi_epi32(r8,r9); // 130 ...
    ta = _mm512_unpacklo_epi32(ra,rb); // 160 ...
    tb = _mm512_unpackhi_epi32(ra,rb); // 162 ...
    tc = _mm512_unpacklo_epi32(rc,rd); // 196 ...
    td = _mm512_unpackhi_epi32(rc,rd); // 198 ...
    te = _mm512_unpacklo_epi32(re,rf); // 228 ...
    tf = _mm512_unpackhi_epi32(re,rf); // 230 ...

    r0 = _mm512_unpacklo_epi64(t0,t2); //   0  16  32  48 ...
    r1 = _mm512_unpackhi_epi64(t0,t2); //   1  17  33  49 ...
    r2 = _mm512_unpacklo_epi64(t1,t3); //   2  18  34  49 ...
    r3 = _mm512_unpackhi_epi64(t1,t3); //   3  19  35  51 ...
    r4 = _mm512_unpacklo_epi64(t4,t6); //  64  80  96 112 ...
    r5 = _mm512_unpackhi_epi64(t4,t6); //  65  81  97 114 ...
    r6 = _mm512_unpacklo_epi64(t5,t7); //  66  82  98 113 ...
    r7 = _mm512_unpackhi_epi64(t5,t7); //  67  83  99 115 ...
    r8 = _mm512_unpacklo_epi64(t8,ta); // 128 144 160 176 ...
    r9 = _mm512_unpackhi_epi64(t8,ta); // 129 145 161 178 ...
    ra = _mm512_unpacklo_epi64(t9,tb); // 130 146 162 177 ...
    rb = _mm512_unpackhi_epi64(t9,tb); // 131 147 163 179 ...
    rc = _mm512_unpacklo_epi64(tc,te); // 192 208 228 240 ...
    rd = _mm512_unpackhi_epi64(tc,te); // 193 209 229 241 ...
    re = _mm512_unpacklo_epi64(td,tf); // 194 210 230 242 ...
    rf = _mm512_unpackhi_epi64(td,tf); // 195 211 231 243 ...

    t0 = _mm512_shuffle_i32x4(r0, r4, 0x88); //   0  16  32  48   8  24  40  56  64  80  96  112 ...
    t1 = _mm512_shuffle_i32x4(r1, r5, 0x88); //   1  17  33  49 ...
    t2 = _mm512_shuffle_i32x4(r2, r6, 0x88); //   2  18  34  50 ...
    t3 = _mm512_shuffle_i32x4(r3, r7, 0x88); //   3  19  35  51 ...
    t4 = _mm512_shuffle_i32x4(r0, r4, 0xdd); //   4  20  36  52 ...
    t5 = _mm512_shuffle_i32x4(r1, r5, 0xdd); //   5  21  37  53 ...
    t6 = _mm512_shuffle_i32x4(r2, r6, 0xdd); //   6  22  38  54 ...
    t7 = _mm512_shuffle_i32x4(r3, r7, 0xdd); //   7  23  39  55 ...
    t8 = _mm512_shuffle_i32x4(r8, rc, 0x88); // 128 144 160 176 ...
    t9 = _mm512_shuffle_i32x4(r9, rd, 0x88); // 129 145 161 177 ...
    ta = _mm512_shuffle_i32x4(ra, re, 0x88); // 130 146 162 178 ...
    tb = _mm512_shuffle_i32x4(rb, rf, 0x88); // 131 147 163 179 ...
    tc = _mm512_shuffle_i32x4(r8, rc, 0xdd); // 132 148 164 180 ...
    td = _mm512_shuffle_i32x4(r9, rd, 0xdd); // 133 149 165 181 ...
    te = _mm512_shuffle_i32x4(ra, re, 0xdd); // 134 150 166 182 ...
    tf = _mm512_shuffle_i32x4(rb, rf, 0xdd); // 135 151 167 183 ...

    r0 = _mm512_shuffle_i32x4(t0, t8, 0x88); //   0  16  32  48  64  80  96 112 ... 240
    r1 = _mm512_shuffle_i32x4(t1, t9, 0x88); //   1  17  33  49  66  81  97 113 ... 241
    r2 = _mm512_shuffle_i32x4(t2, ta, 0x88); //   2  18  34  50  67  82  98 114 ... 242
    r3 = _mm512_shuffle_i32x4(t3, tb, 0x88); //   3  19  35  51  68  83  99 115 ... 243
    r4 = _mm512_shuffle_i32x4(t4, tc, 0x88); //   4 ...
    r5 = _mm512_shuffle_i32x4(t5, td, 0x88); //   5 ...
    r6 = _mm512_shuffle_i32x4(t6, te, 0x88); //   6 ...
    r7 = _mm512_shuffle_i32x4(t7, tf, 0x88); //   7 ...
    r8 = _mm512_shuffle_i32x4(t0, t8, 0xdd); //   8 ...
    r9 = _mm512_shuffle_i32x4(t1, t9, 0xdd); //   9 ...
    ra = _mm512_shuffle_i32x4(t2, ta, 0xdd); //  10 ...
    rb = _mm512_shuffle_i32x4(t3, tb, 0xdd); //  11 ...
    rc = _mm512_shuffle_i32x4(t4, tc, 0xdd); //  12 ...
    rd = _mm512_shuffle_i32x4(t5, td, 0xdd); //  13 ...
    re = _mm512_shuffle_i32x4(t6, te, 0xdd); //  14 ...
    rf = _mm512_shuffle_i32x4(t7, tf, 0xdd); //  15  31  47  63  79  96 111 127 ... 255

    r0f = _mm512_castsi512_ps(r0);
    r1f = _mm512_castsi512_ps(r1);
    r2f = _mm512_castsi512_ps(r2);
    r3f = _mm512_castsi512_ps(r3);
    r4f = _mm512_castsi512_ps(r4);
    r5f = _mm512_castsi512_ps(r5);
    r6f = _mm512_castsi512_ps(r6);
    r7f = _mm512_castsi512_ps(r7);
    r8f = _mm512_castsi512_ps(r8);
    r9f = _mm512_castsi512_ps(r9);
    raf = _mm512_castsi512_ps(ra);
    rbf = _mm512_castsi512_ps(rb);
    rcf = _mm512_castsi512_ps(rc);
    rdf = _mm512_castsi512_ps(rd);
    ref = _mm512_castsi512_ps(re);
    rff = _mm512_castsi512_ps(rf);
}

struct Vec16 {
    using VecType = Vec16;
    __m512 value;
    VecType operator+(const VecType& lr) {
        VecType dst = { _mm512_add_ps(value, lr.value) };
        return dst;
    }
    VecType operator-(const VecType& lr) {
        VecType dst = { _mm512_sub_ps(value, lr.value) };
        return dst;
    }
    VecType operator*(const VecType& lr) {
        VecType dst = { _mm512_mul_ps(value, lr.value) };
        return dst;
    }
    VecType operator*(float lr) {
        VecType dst = { _mm512_mul_ps(value, _mm512_set1_ps(lr)) };
        return dst;
    }
    VecType operator+=(const VecType& lr) {
        value = _mm512_add_ps(value, lr.value);
        return *this;
    }
    VecType operator-=(const VecType& lr) {
        value = _mm512_sub_ps(value, lr.value);
        return *this;
    }

    VecType& operator=(const VecType& lr) {
        value = lr.value;
        return *this;
    }
    VecType operator-() {
        VecType dst;
#if defined(_MSC_VER)
        dst.value = _mm512_xor_ps(value, _mm512_set1_ps(-0.f)); // Using unary operation to SSE vec is GCC extension. We can not do this directly in MSVC.
#else
        dst.value = -value;
#endif
        return dst;
    }
    Vec16() {
    }
    Vec16(const float v) {
        value = _mm512_set1_ps(v);
    }
    Vec16(__m512&& v) {
        value = v;
    }
    Vec16(const VecType& lr) {
        value = lr.value;
    }
    float operator[](size_t i) {
#if defined(_MSC_VER)  // X64 native only mandatory support SSE and SSE2 extension, and we can not find intrinsic function to extract element directly by index in SSE and SSE2 extension.
        float temp[16];
        _mm512_storeu_ps(temp, value);
        return temp[i];
#else
        return value[i];
#endif
    }
    VecType operator==(const VecType& lr) const {
        __m512 one = _mm512_set1_ps(1.0f);
        __m512 zero = _mm512_set1_ps(0.0f);
        __mmask16 mask = _mm512_cmp_ps_mask(value, lr.value, 0);
        VecType dst =  { _mm512_mask_blend_ps(mask, zero, one) } ;
        return dst;
    }
    VecType operator>(const VecType& lr) {
        __m512 one = _mm512_set1_ps(1.0f);
        __m512 zero = _mm512_set1_ps(0.0f);
        __mmask16 mask = _mm512_cmp_ps_mask(value, lr.value, 14);
         VecType dst =  { _mm512_mask_blend_ps(mask, zero, one) } ;
        return dst;
    }
    VecType operator>=(const VecType& lr) {
        __m512 one = _mm512_set1_ps(1.0f);
        __m512 zero = _mm512_set1_ps(0.0f);
        __mmask16 mask = _mm512_cmp_ps_mask(value, lr.value, 13);
        VecType dst =  { _mm512_mask_blend_ps(mask, zero, one) } ;
        return dst;
    }
    VecType operator<(const VecType& lr) {
        __m512 one = _mm512_set1_ps(1.0f);
        __m512 zero = _mm512_set1_ps(0.0f);
        __mmask16 mask = _mm512_cmp_ps_mask(value, lr.value, 0x01);
        VecType dst =  { _mm512_mask_blend_ps(mask, zero, one) } ;
        return dst;
    }
    VecType operator<=(const VecType& lr) {
        __m512 one = _mm512_set1_ps(1.0f);
        __m512 zero = _mm512_set1_ps(0.0f);
        __mmask16 mask = _mm512_cmp_ps_mask(value, lr.value, 0x02);
        VecType dst =  { _mm512_mask_blend_ps(mask, zero, one) } ;
        return dst;
    }
    static VecType load(const float* addr) {
        VecType v = { _mm512_loadu_ps(addr) };
        return v;
    }
    static VecType broadcast(const float* addr) {
        VecType dst = { _mm512_set1_ps(*addr) }; // compiled into 'vbroadcastss'
        return dst;
    }
    static void save(float* addr, const VecType& v) {
        _mm512_storeu_ps(addr, v.value);
    }
    static void save(int32_t* addr, const VecType& v) {
        _mm512_storeu_ps((float*)addr, v.value);
    }
    static VecType max(const VecType& v1, const VecType& v2) {
        VecType dst = { _mm512_max_ps(v1.value, v2.value) };
        return dst;
    }
    static VecType min(const VecType& v1, const VecType& v2) {
        VecType dst = { _mm512_min_ps(v1.value, v2.value) };
        return dst;
    }
    static VecType fma(const VecType& v0, const VecType& v1, const VecType& v2) {
        VecType dst = { _mm512_fmadd_ps(v1.value, v2.value, v0.value) };
        return dst;
    }
    static VecType fms(const VecType& v0, const VecType& v1, const VecType& v2) {
        VecType dst = { _mm512_fnmadd_ps(v1.value, v2.value, v0.value) };
        return dst;
    }

    static void inline transpose16(VecType& r0, VecType& r1, VecType& r2, VecType& r3, VecType& r4, VecType& r5,
                                      VecType& r6, VecType& r7, VecType& r8, VecType& r9, VecType& ra, VecType& rb,
                                      VecType& rc, VecType& rd, VecType& re, VecType& rf) {
        transpose16x16F(r0.value, r1.value, r2.value, r3.value, r4.value, r5.value, r6.value, r7.value, r8.value,
                       r9.value, ra.value, rb.value, rc.value, rd.value, re.value, rf.value);
    }
};

#endif
