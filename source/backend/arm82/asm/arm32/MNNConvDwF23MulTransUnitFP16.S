//
//  MNNConvDwF23MulTransUnitFP16.S
//  MNN
//
//  Created by MNN on 2019/4/4.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifdef __arm__
#ifndef __aarch64__

#include "MNNAsmGlobal.h"

.text
.align 5

asm_function MNNConvDwF23MulTransUnitFP16
//void MNNConvDwF23MulTransUnitFP16(FLOAT16 **cacheLine, const FLOAT16 *weigth, FLOAT16 *dest, size_t ow);
//Auto: r0:cacheLine, r1:weight, r2:dest, r3:ow
push {r4-r11, lr}
ldr r8, [sp, #36] // biasPtr
ldr r9, [sp, #40] // postParameters
ldr r10, [r9, #8] // minF
ldr r11, [r9, #12] // maxF

vpush {q4-q7}
ldr r4, [r0, #0]
ldr r5, [r0, #4]
ldr r6, [r0, #8]

vld1.16 {q4, q5}, [r1]!
vld1.16 {q6, q7}, [r1]!
vld1.16 {q8, q9}, [r1]!

L2:
cmp r3, #2
blt L1

LoopL2:
mov r7, r1

vld1.16 {q12, q13}, [r4]!
vmul.f16 q0, q4, q12
vld1.16 {q14, q15}, [r4]!
vmul.f16 q1, q5, q13
vld1.16 {q10, q11}, [r7]!
vmul.f16 q2, q6, q14
vld1.16 {q12, q13}, [r5]!
vmul.f16 q3, q7, q15

vmla.f16 q0, q8, q12
vld1.16 {q14, q15}, [r5]!
vmla.f16 q1, q9, q13
vmla.f16 q2, q10, q14
vmla.f16 q3, q11, q15

vld1.16 {q10, q11}, [r7]!
vld1.16 {q12, q13}, [r6]!
vmla.f16 q0, q10, q12
vmla.f16 q1, q11, q13
vld1.16 {q10, q11}, [r7]!
vadd.f16 q0, q1, q0
vld1.16 {q14, q15}, [r6]!

vmla.f16 q2, q10, q14
vmla.f16 q3, q11, q15
vadd.f16 q0, q0, q2

vadd.f16 q3, q3, q1
vsub.f16 q1, q3, q2

vld1.32 {q10}, [r8]
vdup.32 q11, r10
vdup.32 q12, r11
vcvt.f16.f32 d22, q11
vcvt.f16.f32 d24, q12
vmov.32 d23, d22
vmov.32 d25, d24

vadd.f16 q0, q10, q0
vadd.f16 q1, q10, q1

vmin.f16 q0, q12, q0
vmin.f16 q1, q12, q1

vmax.f16 q0, q11, q0
vmax.f16 q1, q11, q1


vst1.16 {q0, q1}, [r2]!

sub r3, r3, #2
cmp r3, #2
bge LoopL2


L1:
cmp r3, #0
beq End
mov r7, r1
mov r12, #32
vld1.16 {q12, q13}, [r4]!
vmul.f16 q0, q4, q12
vld1.16 {q14}, [r4]!
vmul.f16 q1, q5, q13
vld1.16 {q10}, [r7], r12
vmul.f16 q2, q6, q14
vld1.16 {q12, q13}, [r5]!

vmla.f16 q0, q8, q12
vld1.16 {q14}, [r5]!
vmla.f16 q1, q9, q13
vmla.f16 q2, q10, q14

vld1.16 {q10, q11}, [r7]!
vld1.16 {q12, q13}, [r6]!
vmla.f16 q0, q10, q12
vmla.f16 q1, q11, q13
vld1.16 {q10}, [r7]
vld1.16 {q14}, [r6]!

vmla.f16 q2, q10, q14

vadd.f16 q0, q1, q0
vadd.f16 q0, q0, q2

vld1.32 {q10}, [r8]
vdup.32 q11, r10
vdup.32 q12, r11
vcvt.f16.f32 d22, q11
vcvt.f16.f32 d24, q12
vmov.32 d23, d22
vmov.32 d25, d24

vadd.f16 q0, q10, q0

vmin.f16 q0, q12, q0

vmax.f16 q0, q11, q0

vst1.16 {q0}, [r2]!
End:

vpop {q4-q7}
pop {r4-r11, pc}

#endif
#endif
