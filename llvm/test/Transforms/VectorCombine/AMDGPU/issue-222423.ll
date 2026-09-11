; REQUIRES: amdgpu-registered-target
; RUN: opt -S -passes='slp-vectorizer,vector-combine' < %s | FileCheck %s
; RUN: opt -S -passes=vector-combine < %s | FileCheck %s --check-prefixes=ROOTLESS,DEFAULT
; RUN: opt -S -passes=vector-combine -vector-combine-max-scan-instrs=1 < %s | FileCheck %s --check-prefix=LOW-BUDGET
; RUN: opt -S -passes='debugify,vector-combine' < %s | FileCheck %s --check-prefix=DEBUG

target triple = "amdgcn-amd-amdhsa"

declare void @side_effect()
declare void @convergent_barrier() #2
declare void @use_narrow(<2 x float>, <2 x float>, <2 x float>,
                         <2 x float>)
declare void @use_narrow_i32(<2 x i32>, <2 x i32>)

; The gfx950 cost model deliberately lets SLP form four v2f32 trees. Check that
; VectorCombine joins the complete group even without a concatenating root.

; CHECK-LABEL: define void @direct(<8 x float> %a, <8 x float> %b, <8 x float> %c, <8 x float> %d, <8 x i1> %cond, ptr addrspace(3) %out0, ptr addrspace(3) %out1)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[SUB:%.*]] = fsub <8 x float> splat (float 1.000000e+00), %a
; CHECK-NEXT:    [[MUL0:%.*]] = fmul <8 x float> %b, [[SUB]]
; CHECK-NEXT:    [[ADD:%.*]] = fadd <8 x float> %c, [[MUL0]]
; CHECK-NEXT:    [[MUL1:%.*]] = fmul <8 x float> %d, [[ADD]]
; CHECK-NEXT:    [[SELECT:%.*]] = select <8 x i1> %cond, <8 x float> [[MUL1]], <8 x float> zeroinitializer
; CHECK-NEXT:    [[SLICE0:%.*]] = shufflevector <8 x float> [[SELECT]], <8 x float> poison, <2 x i32> <i32 0, i32 1>
; CHECK-NEXT:    [[SLICE1:%.*]] = shufflevector <8 x float> [[SELECT]], <8 x float> poison, <2 x i32> <i32 2, i32 3>
; CHECK-NEXT:    [[SLICE2:%.*]] = shufflevector <8 x float> [[SELECT]], <8 x float> poison, <2 x i32> <i32 4, i32 5>
; CHECK-NEXT:    [[SLICE3:%.*]] = shufflevector <8 x float> [[SELECT]], <8 x float> poison, <2 x i32> <i32 6, i32 7>
; CHECK-NEXT:    [[BF0:%.*]] = fptrunc <2 x float> [[SLICE0]] to <2 x bfloat>
; CHECK-NEXT:    [[BF1:%.*]] = fptrunc <2 x float> [[SLICE1]] to <2 x bfloat>
; CHECK-NEXT:    [[BF2:%.*]] = fptrunc <2 x float> [[SLICE2]] to <2 x bfloat>
; CHECK-NEXT:    [[BF3:%.*]] = fptrunc <2 x float> [[SLICE3]] to <2 x bfloat>
; CHECK-NEXT:    [[LO:%.*]] = shufflevector <2 x bfloat> [[BF0]], <2 x bfloat> [[BF1]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT:    store <4 x bfloat> [[LO]], ptr addrspace(3) %out0, align 8
; CHECK-NEXT:    [[HI:%.*]] = shufflevector <2 x bfloat> [[BF2]], <2 x bfloat> [[BF3]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT:    store <4 x bfloat> [[HI]], ptr addrspace(3) %out1, align 8
; CHECK-NEXT:    ret void
define void @direct(<8 x float> %a, <8 x float> %b, <8 x float> %c,
                    <8 x float> %d, <8 x i1> %cond,
                    ptr addrspace(3) %out0, ptr addrspace(3) %out1) #0 {
entry:
  %a0 = extractelement <8 x float> %a, i32 0
  %a1 = extractelement <8 x float> %a, i32 1
  %a2 = extractelement <8 x float> %a, i32 2
  %a3 = extractelement <8 x float> %a, i32 3
  %a4 = extractelement <8 x float> %a, i32 4
  %a5 = extractelement <8 x float> %a, i32 5
  %a6 = extractelement <8 x float> %a, i32 6
  %a7 = extractelement <8 x float> %a, i32 7
  %b0 = extractelement <8 x float> %b, i32 0
  %b1 = extractelement <8 x float> %b, i32 1
  %b2 = extractelement <8 x float> %b, i32 2
  %b3 = extractelement <8 x float> %b, i32 3
  %b4 = extractelement <8 x float> %b, i32 4
  %b5 = extractelement <8 x float> %b, i32 5
  %b6 = extractelement <8 x float> %b, i32 6
  %b7 = extractelement <8 x float> %b, i32 7
  %c0 = extractelement <8 x float> %c, i32 0
  %c1 = extractelement <8 x float> %c, i32 1
  %c2 = extractelement <8 x float> %c, i32 2
  %c3 = extractelement <8 x float> %c, i32 3
  %c4 = extractelement <8 x float> %c, i32 4
  %c5 = extractelement <8 x float> %c, i32 5
  %c6 = extractelement <8 x float> %c, i32 6
  %c7 = extractelement <8 x float> %c, i32 7
  %d0 = extractelement <8 x float> %d, i32 0
  %d1 = extractelement <8 x float> %d, i32 1
  %d2 = extractelement <8 x float> %d, i32 2
  %d3 = extractelement <8 x float> %d, i32 3
  %d4 = extractelement <8 x float> %d, i32 4
  %d5 = extractelement <8 x float> %d, i32 5
  %d6 = extractelement <8 x float> %d, i32 6
  %d7 = extractelement <8 x float> %d, i32 7
  %p0 = extractelement <8 x i1> %cond, i32 0
  %p1 = extractelement <8 x i1> %cond, i32 1
  %p2 = extractelement <8 x i1> %cond, i32 2
  %p3 = extractelement <8 x i1> %cond, i32 3
  %p4 = extractelement <8 x i1> %cond, i32 4
  %p5 = extractelement <8 x i1> %cond, i32 5
  %p6 = extractelement <8 x i1> %cond, i32 6
  %p7 = extractelement <8 x i1> %cond, i32 7
  %s0 = fsub float 1.0, %a0
  %s1 = fsub float 1.0, %a1
  %s2 = fsub float 1.0, %a2
  %s3 = fsub float 1.0, %a3
  %s4 = fsub float 1.0, %a4
  %s5 = fsub float 1.0, %a5
  %s6 = fsub float 1.0, %a6
  %s7 = fsub float 1.0, %a7
  %m0 = fmul float %b0, %s0
  %m1 = fmul float %b1, %s1
  %m2 = fmul float %b2, %s2
  %m3 = fmul float %b3, %s3
  %m4 = fmul float %b4, %s4
  %m5 = fmul float %b5, %s5
  %m6 = fmul float %b6, %s6
  %m7 = fmul float %b7, %s7
  %x0 = fadd float %c0, %m0
  %x1 = fadd float %c1, %m1
  %x2 = fadd float %c2, %m2
  %x3 = fadd float %c3, %m3
  %x4 = fadd float %c4, %m4
  %x5 = fadd float %c5, %m5
  %x6 = fadd float %c6, %m6
  %x7 = fadd float %c7, %m7
  %y0 = fmul float %d0, %x0
  %y1 = fmul float %d1, %x1
  %y2 = fmul float %d2, %x2
  %y3 = fmul float %d3, %x3
  %y4 = fmul float %d4, %x4
  %y5 = fmul float %d5, %x5
  %y6 = fmul float %d6, %x6
  %y7 = fmul float %d7, %x7
  %z0 = select i1 %p0, float %y0, float 0.0
  %z1 = select i1 %p1, float %y1, float 0.0
  %z2 = select i1 %p2, float %y2, float 0.0
  %z3 = select i1 %p3, float %y3, float 0.0
  %z4 = select i1 %p4, float %y4, float 0.0
  %z5 = select i1 %p5, float %y5, float 0.0
  %z6 = select i1 %p6, float %y6, float 0.0
  %z7 = select i1 %p7, float %y7, float 0.0
  %pair01.0 = insertelement <2 x float> undef, float %z0, i32 0
  %pair01 = insertelement <2 x float> %pair01.0, float %z1, i32 1
  %bf01 = fptrunc <2 x float> %pair01 to <2 x bfloat>
  %pair23.0 = insertelement <2 x float> undef, float %z2, i32 0
  %pair23 = insertelement <2 x float> %pair23.0, float %z3, i32 1
  %bf23 = fptrunc <2 x float> %pair23 to <2 x bfloat>
  %pair45.0 = insertelement <2 x float> undef, float %z4, i32 0
  %pair45 = insertelement <2 x float> %pair45.0, float %z5, i32 1
  %bf45 = fptrunc <2 x float> %pair45 to <2 x bfloat>
  %pair67.0 = insertelement <2 x float> undef, float %z6, i32 0
  %pair67 = insertelement <2 x float> %pair67.0, float %z7, i32 1
  %bf67 = fptrunc <2 x float> %pair67 to <2 x bfloat>
  %lo = shufflevector <2 x bfloat> %bf01, <2 x bfloat> %bf23,
                      <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x bfloat> %lo, ptr addrspace(3) %out0, align 8
  %hi = shufflevector <2 x bfloat> %bf45, <2 x bfloat> %bf67,
                      <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x bfloat> %hi, ptr addrspace(3) %out1, align 8
  ret void
}

; Keep a concise integration check for the asm-seeded form of the issue. The
; tied asm and its full-width data flow must remain after VectorCombine.

; CHECK-LABEL: define void @asm_seeded(<8 x float> %a, <8 x float> %b, ptr addrspace(3) %out)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ADD:%.*]] = fadd <8 x float> %a, %b
; CHECK-NEXT:    [[BITS:%.*]] = bitcast <8 x float> [[ADD]] to <8 x i32>
; CHECK-NEXT:    [[BAR:%.*]] = call <8 x i32> asm "", "=v,0"(<8 x i32> [[BITS]])
; CHECK-NEXT:    [[RESULT:%.*]] = bitcast <8 x i32> [[BAR]] to <8 x float>
; CHECK-NEXT:    store <8 x float> [[RESULT]], ptr addrspace(3) %out, align 32
; CHECK-NEXT:    ret void
define void @asm_seeded(<8 x float> %a, <8 x float> %b,
                        ptr addrspace(3) %out) #0 {
entry:
  %a01 = shufflevector <8 x float> %a, <8 x float> poison,
                       <2 x i32> <i32 0, i32 1>
  %b01 = shufflevector <8 x float> %b, <8 x float> poison,
                       <2 x i32> <i32 0, i32 1>
  %add01 = fadd <2 x float> %a01, %b01
  %a23 = shufflevector <8 x float> %a, <8 x float> poison,
                       <2 x i32> <i32 2, i32 3>
  %b23 = shufflevector <8 x float> %b, <8 x float> poison,
                       <2 x i32> <i32 2, i32 3>
  %add23 = fadd <2 x float> %a23, %b23
  %a45 = shufflevector <8 x float> %a, <8 x float> poison,
                       <2 x i32> <i32 4, i32 5>
  %b45 = shufflevector <8 x float> %b, <8 x float> poison,
                       <2 x i32> <i32 4, i32 5>
  %add45 = fadd <2 x float> %a45, %b45
  %a67 = shufflevector <8 x float> %a, <8 x float> poison,
                       <2 x i32> <i32 6, i32 7>
  %b67 = shufflevector <8 x float> %b, <8 x float> poison,
                       <2 x i32> <i32 6, i32 7>
  %add67 = fadd <2 x float> %a67, %b67
  %lo = shufflevector <2 x float> %add01, <2 x float> %add23,
                      <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %hi = shufflevector <2 x float> %add45, <2 x float> %add67,
                      <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %wide = shufflevector <4 x float> %lo, <4 x float> %hi,
                        <8 x i32> <i32 0, i32 1, i32 2, i32 3,
                                    i32 4, i32 5, i32 6, i32 7>
  %bits = bitcast <8 x float> %wide to <8 x i32>
  %bar = call <8 x i32> asm "", "=v,0"(<8 x i32> %bits)
  %result = bitcast <8 x i32> %bar to <8 x float>
  store <8 x float> %result, ptr addrspace(3) %out, align 32
  ret void
}

; Exercise the post-SLP fold directly, independently of SLP's current VF
; choice and without a concat user.

; Sinking the replacement slices must not create forward references in debug
; records that described the original roots.
; DEBUG-LABEL: define void @rootless_v2(
; DEBUG-NOT:     #dbg_value(<2 x float> %select{{(01|23|45|67)}},
; DEBUG:         %select01 = shufflevector
; DEBUG-NOT:     #dbg_value(<2 x float> %select{{(23|45|67)}},
; DEBUG:         %select23 = shufflevector
; DEBUG-NOT:     #dbg_value(<2 x float> %select{{(45|67)}},
; DEBUG:         %select45 = shufflevector
; DEBUG-NOT:     #dbg_value(<2 x float> %select67,
; DEBUG:         %select67 = shufflevector

; LOW-BUDGET-LABEL: define void @rootless_v2(
; LOW-BUDGET:         fadd <2 x float>
; LOW-BUDGET:         select <2 x i1>
; LOW-BUDGET:         fadd <2 x float>
; LOW-BUDGET:         select <2 x i1>
; LOW-BUDGET:         fadd <2 x float>
; LOW-BUDGET:         select <2 x i1>
; LOW-BUDGET:         fadd <2 x float>
; LOW-BUDGET:         select <2 x i1>
; LOW-BUDGET-NOT:     fadd <8 x float>
; LOW-BUDGET-NOT:     select <8 x i1>
; LOW-BUDGET:         ret void

; ROOTLESS-LABEL: define void @rootless_v2(<8 x float> %a, <8 x float> %b, <8 x i1> %cond, ptr addrspace(3) %out0, ptr addrspace(3) %out1, ptr addrspace(3) %out2, ptr addrspace(3) %out3)
; ROOTLESS-NEXT:  entry:
; ROOTLESS-NEXT:    [[ADD:%.*]] = fadd <8 x float> %a, %b
; ROOTLESS-NEXT:    [[SELECT:%.*]] = select <8 x i1> %cond, <8 x float> [[ADD]], <8 x float> zeroinitializer
; ROOTLESS-NEXT:    [[SLICE0:%.*]] = shufflevector <8 x float> [[SELECT]], <8 x float> poison, <2 x i32> <i32 0, i32 1>
; ROOTLESS-NEXT:    [[SLICE1:%.*]] = shufflevector <8 x float> [[SELECT]], <8 x float> poison, <2 x i32> <i32 2, i32 3>
; ROOTLESS-NEXT:    [[SLICE2:%.*]] = shufflevector <8 x float> [[SELECT]], <8 x float> poison, <2 x i32> <i32 4, i32 5>
; ROOTLESS-NEXT:    [[SLICE3:%.*]] = shufflevector <8 x float> [[SELECT]], <8 x float> poison, <2 x i32> <i32 6, i32 7>
; ROOTLESS-NEXT:    store <2 x float> [[SLICE0]], ptr addrspace(3) %out0, align 8
; ROOTLESS-NEXT:    store <2 x float> [[SLICE1]], ptr addrspace(3) %out1, align 8
; ROOTLESS-NEXT:    store <2 x float> [[SLICE2]], ptr addrspace(3) %out2, align 8
; ROOTLESS-NEXT:    store <2 x float> [[SLICE3]], ptr addrspace(3) %out3, align 8
; ROOTLESS-NEXT:    ret void
define void @rootless_v2(<8 x float> %a, <8 x float> %b, <8 x i1> %cond,
                         ptr addrspace(3) %out0, ptr addrspace(3) %out1,
                         ptr addrspace(3) %out2, ptr addrspace(3) %out3) #0 {
entry:
  %a01 = shufflevector <8 x float> %a, <8 x float> poison,
                       <2 x i32> <i32 0, i32 1>
  %b01 = shufflevector <8 x float> %b, <8 x float> poison,
                       <2 x i32> <i32 0, i32 1>
  %add01 = fadd <2 x float> %a01, %b01
  %p01 = shufflevector <8 x i1> %cond, <8 x i1> poison,
                       <2 x i32> <i32 0, i32 1>
  %select01 = select <2 x i1> %p01, <2 x float> %add01,
                     <2 x float> zeroinitializer
  %a23 = shufflevector <8 x float> %a, <8 x float> poison,
                       <2 x i32> <i32 2, i32 3>
  %b23 = shufflevector <8 x float> %b, <8 x float> poison,
                       <2 x i32> <i32 2, i32 3>
  %add23 = fadd <2 x float> %a23, %b23
  %p23 = shufflevector <8 x i1> %cond, <8 x i1> poison,
                       <2 x i32> <i32 2, i32 3>
  %select23 = select <2 x i1> %p23, <2 x float> %add23,
                     <2 x float> zeroinitializer
  %a45 = shufflevector <8 x float> %a, <8 x float> poison,
                       <2 x i32> <i32 4, i32 5>
  %b45 = shufflevector <8 x float> %b, <8 x float> poison,
                       <2 x i32> <i32 4, i32 5>
  %add45 = fadd <2 x float> %a45, %b45
  %p45 = shufflevector <8 x i1> %cond, <8 x i1> poison,
                       <2 x i32> <i32 4, i32 5>
  %select45 = select <2 x i1> %p45, <2 x float> %add45,
                     <2 x float> zeroinitializer
  %a67 = shufflevector <8 x float> %a, <8 x float> poison,
                       <2 x i32> <i32 6, i32 7>
  %b67 = shufflevector <8 x float> %b, <8 x float> poison,
                       <2 x i32> <i32 6, i32 7>
  %add67 = fadd <2 x float> %a67, %b67
  %p67 = shufflevector <8 x i1> %cond, <8 x i1> poison,
                       <2 x i32> <i32 6, i32 7>
  %select67 = select <2 x i1> %p67, <2 x float> %add67,
                     <2 x float> zeroinitializer
  store <2 x float> %select01, ptr addrspace(3) %out0, align 8
  store <2 x float> %select23, ptr addrspace(3) %out1, align 8
  store <2 x float> %select45, ptr addrspace(3) %out2, align 8
  store <2 x float> %select67, ptr addrspace(3) %out3, align 8
  ret void
}

; Reject extracts from operand 1 and masks with poison lanes. Treating either
; as an all-defined operand-0 slice would incorrectly define poison values.

; DEFAULT-LABEL: define void @noncanonical_extract_masks(
; DEFAULT:         [[SECOND_P_LO:%.*]] = shufflevector <4 x i1> %p, <4 x i1> poison, <2 x i32> <i32 4, i32 5>
; DEFAULT:         select <2 x i1> [[SECOND_P_LO]]
; DEFAULT:         [[SECOND_P_HI:%.*]] = shufflevector <4 x i1> %p, <4 x i1> poison, <2 x i32> <i32 6, i32 7>
; DEFAULT:         select <2 x i1> [[SECOND_P_HI]]
; DEFAULT:         [[POISON_Q_LO:%.*]] = shufflevector <4 x i1> %q, <4 x i1> poison, <2 x i32> <i32 0, i32 poison>
; DEFAULT:         select <2 x i1> [[POISON_Q_LO]]
; DEFAULT:         [[POISON_Q_HI:%.*]] = shufflevector <4 x i1> %q, <4 x i1> poison, <2 x i32> <i32 2, i32 poison>
; DEFAULT:         select <2 x i1> [[POISON_Q_HI]]
; DEFAULT-NOT:     select <4 x i1>
; DEFAULT:         call void @use_narrow
; DEFAULT:         ret void
define void @noncanonical_extract_masks(<4 x float> %a, <4 x float> %b,
                                        <4 x float> %c, <4 x float> %d,
                                        <4 x i1> %p, <4 x i1> %q) #0 {
entry:
  %second.p.lo = shufflevector <4 x i1> %p, <4 x i1> poison,
                               <2 x i32> <i32 4, i32 5>
  %second.a.lo = shufflevector <4 x float> %a, <4 x float> poison,
                               <2 x i32> <i32 4, i32 5>
  %second.c.lo = shufflevector <4 x float> %c, <4 x float> poison,
                               <2 x i32> <i32 4, i32 5>
  %second.add.lo = fadd <2 x float> %second.a.lo, %second.c.lo
  %second.select.lo = select <2 x i1> %second.p.lo,
                             <2 x float> %second.add.lo,
                             <2 x float> zeroinitializer
  %second.p.hi = shufflevector <4 x i1> %p, <4 x i1> poison,
                               <2 x i32> <i32 6, i32 7>
  %second.a.hi = shufflevector <4 x float> %a, <4 x float> poison,
                               <2 x i32> <i32 6, i32 7>
  %second.c.hi = shufflevector <4 x float> %c, <4 x float> poison,
                               <2 x i32> <i32 6, i32 7>
  %second.add.hi = fadd <2 x float> %second.a.hi, %second.c.hi
  %second.select.hi = select <2 x i1> %second.p.hi,
                             <2 x float> %second.add.hi,
                             <2 x float> zeroinitializer
  %poison.q.lo = shufflevector <4 x i1> %q, <4 x i1> poison,
                               <2 x i32> <i32 0, i32 poison>
  %poison.b.lo = shufflevector <4 x float> %b, <4 x float> poison,
                               <2 x i32> <i32 0, i32 poison>
  %poison.d.lo = shufflevector <4 x float> %d, <4 x float> poison,
                               <2 x i32> <i32 0, i32 poison>
  %poison.add.lo = fadd <2 x float> %poison.b.lo, %poison.d.lo
  %poison.select.lo = select <2 x i1> %poison.q.lo,
                             <2 x float> %poison.add.lo,
                             <2 x float> zeroinitializer
  %poison.q.hi = shufflevector <4 x i1> %q, <4 x i1> poison,
                               <2 x i32> <i32 2, i32 poison>
  %poison.b.hi = shufflevector <4 x float> %b, <4 x float> poison,
                               <2 x i32> <i32 2, i32 poison>
  %poison.d.hi = shufflevector <4 x float> %d, <4 x float> poison,
                               <2 x i32> <i32 2, i32 poison>
  %poison.add.hi = fadd <2 x float> %poison.b.hi, %poison.d.hi
  %poison.select.hi = select <2 x i1> %poison.q.hi,
                             <2 x float> %poison.add.hi,
                             <2 x float> zeroinitializer
  call void @use_narrow(<2 x float> %second.select.lo,
                        <2 x float> %second.select.hi,
                        <2 x float> %poison.select.lo,
                        <2 x float> %poison.select.hi)
  ret void
}

; Retain opaque, multi-use leaves and concatenate them in lane order while
; widening the profitable operations above them.

; Synthetic concatenations have no single source location. They must not
; inherit a stale location, while the rebuilt operation retains one.
; DEBUG-LABEL: define void @opaque_multiuse_leaves(
; DEBUG:         [[DBGXLO:%.*]] = shufflevector <2 x float> %x01, <2 x float> %x23, <4 x i32> <i32 0, i32 1, i32 2, i32 3>{{$}}
; DEBUG-NEXT:    [[DBGXHI:%.*]] = shufflevector <2 x float> %x45, <2 x float> %x67, <4 x i32> <i32 0, i32 1, i32 2, i32 3>{{$}}
; DEBUG-NEXT:    [[DBGXWIDE:%.*]] = shufflevector <4 x float> [[DBGXLO]], <4 x float> [[DBGXHI]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>{{$}}
; DEBUG-NEXT:    {{%.*}} = fmul <8 x float> %a, [[DBGXWIDE]], !dbg

; DEFAULT-LABEL: define void @opaque_multiuse_leaves(
; DEFAULT:         [[X01:%.*]] = freeze <2 x float> %x
; DEFAULT:         [[X23:%.*]] = freeze <2 x float> %x
; DEFAULT:         [[X45:%.*]] = freeze <2 x float> %x
; DEFAULT:         [[X67:%.*]] = freeze <2 x float> %x
; DEFAULT:         [[XLO:%.*]] = shufflevector <2 x float> [[X01]], <2 x float> [[X23]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; DEFAULT-NEXT:    [[XHI:%.*]] = shufflevector <2 x float> [[X45]], <2 x float> [[X67]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; DEFAULT-NEXT:    [[XWIDE:%.*]] = shufflevector <4 x float> [[XLO]], <4 x float> [[XHI]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; DEFAULT-NEXT:    [[MUL:%.*]] = fmul <8 x float> %a, [[XWIDE]]
; DEFAULT-NEXT:    [[ADD:%.*]] = fadd <8 x float> [[MUL]], splat (float 1.000000e+00)
; DEFAULT-NEXT:    [[SELECT:%.*]] = select <8 x i1> %cond, <8 x float> [[ADD]], <8 x float> zeroinitializer
; DEFAULT-NEXT:    [[R01:%.*]] = shufflevector <8 x float> [[SELECT]], <8 x float> poison, <2 x i32> <i32 0, i32 1>
; DEFAULT-NEXT:    [[R23:%.*]] = shufflevector <8 x float> [[SELECT]], <8 x float> poison, <2 x i32> <i32 2, i32 3>
; DEFAULT-NEXT:    [[R45:%.*]] = shufflevector <8 x float> [[SELECT]], <8 x float> poison, <2 x i32> <i32 4, i32 5>
; DEFAULT-NEXT:    [[R67:%.*]] = shufflevector <8 x float> [[SELECT]], <8 x float> poison, <2 x i32> <i32 6, i32 7>
; DEFAULT-NEXT:    call void @use_narrow(<2 x float> [[X01]], <2 x float> [[X23]], <2 x float> [[X45]], <2 x float> [[X67]])
; DEFAULT-NEXT:    call void @use_narrow(<2 x float> [[R01]], <2 x float> [[R23]], <2 x float> [[R45]], <2 x float> [[R67]])
; DEFAULT-NEXT:    ret void
define void @opaque_multiuse_leaves(<8 x float> %a, <8 x i1> %cond,
                                    <2 x float> %x) #0 {
entry:
  %x01 = freeze <2 x float> %x
  %a01 = shufflevector <8 x float> %a, <8 x float> poison,
                       <2 x i32> <i32 0, i32 1>
  %m01 = fmul <2 x float> %a01, %x01
  %r01 = fadd <2 x float> %m01, splat (float 1.0)
  %p01 = shufflevector <8 x i1> %cond, <8 x i1> poison,
                       <2 x i32> <i32 0, i32 1>
  %s01 = select <2 x i1> %p01, <2 x float> %r01,
                <2 x float> zeroinitializer
  %x23 = freeze <2 x float> %x
  %a23 = shufflevector <8 x float> %a, <8 x float> poison,
                       <2 x i32> <i32 2, i32 3>
  %m23 = fmul <2 x float> %a23, %x23
  %r23 = fadd <2 x float> %m23, splat (float 1.0)
  %p23 = shufflevector <8 x i1> %cond, <8 x i1> poison,
                       <2 x i32> <i32 2, i32 3>
  %s23 = select <2 x i1> %p23, <2 x float> %r23,
                <2 x float> zeroinitializer
  %x45 = freeze <2 x float> %x
  %a45 = shufflevector <8 x float> %a, <8 x float> poison,
                       <2 x i32> <i32 4, i32 5>
  %m45 = fmul <2 x float> %a45, %x45
  %r45 = fadd <2 x float> %m45, splat (float 1.0)
  %p45 = shufflevector <8 x i1> %cond, <8 x i1> poison,
                       <2 x i32> <i32 4, i32 5>
  %s45 = select <2 x i1> %p45, <2 x float> %r45,
                <2 x float> zeroinitializer
  %x67 = freeze <2 x float> %x
  %a67 = shufflevector <8 x float> %a, <8 x float> poison,
                       <2 x i32> <i32 6, i32 7>
  %m67 = fmul <2 x float> %a67, %x67
  %r67 = fadd <2 x float> %m67, splat (float 1.0)
  %p67 = shufflevector <8 x i1> %cond, <8 x i1> poison,
                       <2 x i32> <i32 6, i32 7>
  %s67 = select <2 x i1> %p67, <2 x float> %r67,
                <2 x float> zeroinitializer
  call void @use_narrow(<2 x float> %x01, <2 x float> %x23,
                        <2 x float> %x45, <2 x float> %x67)
  call void @use_narrow(<2 x float> %s01, <2 x float> %s23,
                        <2 x float> %s45, <2 x float> %s67)
  ret void
}

; Do not sink a sibling tree across inline assembly. Inline assembly may be an
; intentional scheduling barrier even when it has no sideeffect marker.

; CHECK-LABEL: define void @barrier_between_siblings(
; CHECK:         [[LO:%.*]] = fadd <2 x float>
; CHECK-NEXT:    [[BAR:%.*]] = call <2 x i32> asm "", "=v,0"(<2 x i32> %token)
; CHECK:         [[HI:%.*]] = fadd <2 x float>
; CHECK-NOT:     fadd <4 x float>
; CHECK:         store <2 x i32> [[BAR]], ptr addrspace(3) %bar.out
; CHECK:         store <2 x float> [[LO]]
; CHECK:         store <2 x float> [[HI]]
define void @barrier_between_siblings(<4 x float> %a, <4 x float> %b,
                                      <2 x i32> %token,
                                      ptr addrspace(3) %bar.out,
                                      ptr addrspace(3) %out0,
                                      ptr addrspace(3) %out1) #0 {
entry:
  %a.lo = shufflevector <4 x float> %a, <4 x float> poison,
                        <2 x i32> <i32 0, i32 1>
  %b.lo = shufflevector <4 x float> %b, <4 x float> poison,
                        <2 x i32> <i32 0, i32 1>
  %lo = fadd <2 x float> %a.lo, %b.lo
  %bar = call <2 x i32> asm "", "=v,0"(<2 x i32> %token) #1
  %a.hi = shufflevector <4 x float> %a, <4 x float> poison,
                        <2 x i32> <i32 2, i32 3>
  %b.hi = shufflevector <4 x float> %b, <4 x float> poison,
                        <2 x i32> <i32 2, i32 3>
  %hi = fadd <2 x float> %a.hi, %b.hi
  store <2 x i32> %bar, ptr addrspace(3) %bar.out, align 8
  store <2 x float> %lo, ptr addrspace(3) %out0, align 8
  store <2 x float> %hi, ptr addrspace(3) %out1, align 8
  ret void
}

; Side-effecting and convergent calls are also motion barriers.

; DEFAULT-LABEL: define void @call_barriers(
; DEFAULT:         [[ADDLO:%.*]] = fadd <2 x float>
; DEFAULT-NEXT:    call void @side_effect()
; DEFAULT:         [[ADDHI:%.*]] = fadd <2 x float>
; DEFAULT:         [[SUBLO:%.*]] = fsub <2 x float>
; DEFAULT-NEXT:    call void @convergent_barrier()
; DEFAULT:         [[SUBHI:%.*]] = fsub <2 x float>
; DEFAULT-NOT:     fadd <4 x float>
; DEFAULT-NOT:     fsub <4 x float>
; DEFAULT:         call void @use_narrow(<2 x float> [[ADDLO]], <2 x float> [[ADDHI]], <2 x float> [[SUBLO]], <2 x float> [[SUBHI]])
; DEFAULT:         ret void
define void @call_barriers(<4 x float> %a, <4 x float> %b,
                           <4 x float> %c, <4 x float> %d) #0 {
entry:
  %a.lo = shufflevector <4 x float> %a, <4 x float> poison,
                        <2 x i32> <i32 0, i32 1>
  %b.lo = shufflevector <4 x float> %b, <4 x float> poison,
                        <2 x i32> <i32 0, i32 1>
  %add.lo = fadd <2 x float> %a.lo, %b.lo
  call void @side_effect()
  %a.hi = shufflevector <4 x float> %a, <4 x float> poison,
                        <2 x i32> <i32 2, i32 3>
  %b.hi = shufflevector <4 x float> %b, <4 x float> poison,
                        <2 x i32> <i32 2, i32 3>
  %add.hi = fadd <2 x float> %a.hi, %b.hi
  %c.lo = shufflevector <4 x float> %c, <4 x float> poison,
                        <2 x i32> <i32 0, i32 1>
  %d.lo = shufflevector <4 x float> %d, <4 x float> poison,
                        <2 x i32> <i32 0, i32 1>
  %sub.lo = fsub <2 x float> %c.lo, %d.lo
  call void @convergent_barrier() #2
  %c.hi = shufflevector <4 x float> %c, <4 x float> poison,
                        <2 x i32> <i32 2, i32 3>
  %d.hi = shufflevector <4 x float> %d, <4 x float> poison,
                        <2 x i32> <i32 2, i32 3>
  %sub.hi = fsub <2 x float> %c.hi, %d.hi
  call void @use_narrow(<2 x float> %add.lo, <2 x float> %add.hi,
                        <2 x float> %sub.lo, <2 x float> %sub.hi)
  ret void
}

; Require all contiguous chunks before rebuilding a wider tree.

; DEFAULT-LABEL: define void @incomplete_sibling_group(
; DEFAULT-COUNT-3: fadd <2 x float>
; DEFAULT-NOT:     fadd <8 x float>
; DEFAULT:         ret void
define void @incomplete_sibling_group(<8 x float> %a, <8 x float> %b,
                                      ptr addrspace(3) %out0,
                                      ptr addrspace(3) %out1,
                                      ptr addrspace(3) %out3) #0 {
entry:
  %a.01 = shufflevector <8 x float> %a, <8 x float> poison,
                        <2 x i32> <i32 0, i32 1>
  %b.01 = shufflevector <8 x float> %b, <8 x float> poison,
                        <2 x i32> <i32 0, i32 1>
  %add.01 = fadd <2 x float> %a.01, %b.01
  %a.23 = shufflevector <8 x float> %a, <8 x float> poison,
                        <2 x i32> <i32 2, i32 3>
  %b.23 = shufflevector <8 x float> %b, <8 x float> poison,
                        <2 x i32> <i32 2, i32 3>
  %add.23 = fadd <2 x float> %a.23, %b.23
  %a.67 = shufflevector <8 x float> %a, <8 x float> poison,
                        <2 x i32> <i32 6, i32 7>
  %b.67 = shufflevector <8 x float> %b, <8 x float> poison,
                        <2 x i32> <i32 6, i32 7>
  %add.67 = fadd <2 x float> %a.67, %b.67
  store <2 x float> %add.01, ptr addrspace(3) %out0, align 8
  store <2 x float> %add.23, ptr addrspace(3) %out1, align 8
  store <2 x float> %add.67, ptr addrspace(3) %out3, align 8
  ret void
}

; Do not sink a replacement past an earlier surviving use.

; DEFAULT-LABEL: define void @early_sibling_use(
; DEFAULT-COUNT-2: fadd <2 x float>
; DEFAULT-NOT:     fadd <4 x float>
; DEFAULT:         ret void
define void @early_sibling_use(<4 x float> %a, <4 x float> %b,
                               ptr addrspace(3) %out0,
                               ptr addrspace(3) %out1) #0 {
entry:
  %a.lo = shufflevector <4 x float> %a, <4 x float> poison,
                        <2 x i32> <i32 0, i32 1>
  %b.lo = shufflevector <4 x float> %b, <4 x float> poison,
                        <2 x i32> <i32 0, i32 1>
  %lo = fadd <2 x float> %a.lo, %b.lo
  %early = fneg <2 x float> %lo
  %a.hi = shufflevector <4 x float> %a, <4 x float> poison,
                        <2 x i32> <i32 2, i32 3>
  %b.hi = shufflevector <4 x float> %b, <4 x float> poison,
                        <2 x i32> <i32 2, i32 3>
  %hi = fadd <2 x float> %a.hi, %b.hi
  store <2 x float> %early, ptr addrspace(3) %out0, align 8
  store <2 x float> %hi, ptr addrspace(3) %out1, align 8
  ret void
}

; Do not form a replacement tree that retains one of the roots as a leaf.

; DEFAULT-LABEL: define void @dependent_sibling_roots(
; DEFAULT-COUNT-2: fadd <2 x float>
; DEFAULT-NOT:     fadd <4 x float>
; DEFAULT:         ret void
define void @dependent_sibling_roots(<4 x float> %a, <2 x float> %x,
                                     ptr addrspace(3) %out0,
                                     ptr addrspace(3) %out1) #0 {
entry:
  %a.lo = shufflevector <4 x float> %a, <4 x float> poison,
                        <2 x i32> <i32 0, i32 1>
  %lo = fadd <2 x float> %a.lo, %x
  %a.hi = shufflevector <4 x float> %a, <4 x float> poison,
                        <2 x i32> <i32 2, i32 3>
  %hi = fadd <2 x float> %a.hi, %lo
  store <2 x float> %lo, ptr addrspace(3) %out0, align 8
  store <2 x float> %hi, ptr addrspace(3) %out1, align 8
  ret void
}

; Rebuild both groups in the block. Preserve flags common to every sibling and
; drop flags that are not valid for every lane.

; DEFAULT-LABEL: define void @two_folds_and_flags(<4 x float> %a, <4 x float> %b, <4 x i1> %p, <4 x float> %c, <4 x float> %d, <4 x i1> %q)
; DEFAULT-NEXT:  entry:
; DEFAULT-NEXT:    [[ADD:%.*]] = fadd nnan <4 x float> %a, %b
; DEFAULT-NEXT:    [[SELECT0:%.*]] = select <4 x i1> %p, <4 x float> [[ADD]], <4 x float> zeroinitializer
; DEFAULT-NEXT:    [[ADDLO:%.*]] = shufflevector <4 x float> [[SELECT0]], <4 x float> poison, <2 x i32> <i32 0, i32 1>
; DEFAULT-NEXT:    [[ADDHI:%.*]] = shufflevector <4 x float> [[SELECT0]], <4 x float> poison, <2 x i32> <i32 2, i32 3>
; DEFAULT-NEXT:    [[MUL:%.*]] = fmul <4 x float> %c, %d
; DEFAULT-NEXT:    [[SELECT1:%.*]] = select <4 x i1> %q, <4 x float> [[MUL]], <4 x float> zeroinitializer
; DEFAULT-NEXT:    [[MULLO:%.*]] = shufflevector <4 x float> [[SELECT1]], <4 x float> poison, <2 x i32> <i32 0, i32 1>
; DEFAULT-NEXT:    [[MULHI:%.*]] = shufflevector <4 x float> [[SELECT1]], <4 x float> poison, <2 x i32> <i32 2, i32 3>
; DEFAULT-NEXT:    call void @use_narrow(<2 x float> [[ADDLO]], <2 x float> [[ADDHI]], <2 x float> [[MULLO]], <2 x float> [[MULHI]])
; DEFAULT-NEXT:    ret void
define void @two_folds_and_flags(<4 x float> %a, <4 x float> %b,
                                 <4 x i1> %p, <4 x float> %c,
                                 <4 x float> %d, <4 x i1> %q) #0 {
entry:
  %a.lo = shufflevector <4 x float> %a, <4 x float> poison,
                        <2 x i32> <i32 0, i32 1>
  %b.lo = shufflevector <4 x float> %b, <4 x float> poison,
                        <2 x i32> <i32 0, i32 1>
  %p.lo = shufflevector <4 x i1> %p, <4 x i1> poison,
                        <2 x i32> <i32 0, i32 1>
  %add.lo = fadd nnan <2 x float> %a.lo, %b.lo
  %select.add.lo = select <2 x i1> %p.lo, <2 x float> %add.lo,
                          <2 x float> zeroinitializer
  %a.hi = shufflevector <4 x float> %a, <4 x float> poison,
                        <2 x i32> <i32 2, i32 3>
  %b.hi = shufflevector <4 x float> %b, <4 x float> poison,
                        <2 x i32> <i32 2, i32 3>
  %p.hi = shufflevector <4 x i1> %p, <4 x i1> poison,
                        <2 x i32> <i32 2, i32 3>
  %add.hi = fadd nnan <2 x float> %a.hi, %b.hi
  %select.add.hi = select <2 x i1> %p.hi, <2 x float> %add.hi,
                          <2 x float> zeroinitializer
  %c.lo = shufflevector <4 x float> %c, <4 x float> poison,
                        <2 x i32> <i32 0, i32 1>
  %d.lo = shufflevector <4 x float> %d, <4 x float> poison,
                        <2 x i32> <i32 0, i32 1>
  %q.lo = shufflevector <4 x i1> %q, <4 x i1> poison,
                        <2 x i32> <i32 0, i32 1>
  %mul.lo = fmul nnan <2 x float> %c.lo, %d.lo
  %select.mul.lo = select <2 x i1> %q.lo, <2 x float> %mul.lo,
                          <2 x float> zeroinitializer
  %c.hi = shufflevector <4 x float> %c, <4 x float> poison,
                        <2 x i32> <i32 2, i32 3>
  %d.hi = shufflevector <4 x float> %d, <4 x float> poison,
                        <2 x i32> <i32 2, i32 3>
  %q.hi = shufflevector <4 x i1> %q, <4 x i1> poison,
                        <2 x i32> <i32 2, i32 3>
  %mul.hi = fmul <2 x float> %c.hi, %d.hi
  %select.mul.hi = select <2 x i1> %q.hi, <2 x float> %mul.hi,
                          <2 x float> zeroinitializer
  call void @use_narrow(<2 x float> %select.add.lo,
                        <2 x float> %select.add.hi,
                        <2 x float> %select.mul.lo,
                        <2 x float> %select.mul.hi)
  ret void
}

; Do not rebuild a complete sibling group when the target reports equal costs.

; DEFAULT-LABEL: define void @equal_cost_i32(
; DEFAULT-COUNT-2: add <2 x i32>
; DEFAULT-NOT:     add <4 x i32>
; DEFAULT:         call void @use_narrow_i32
; DEFAULT:         ret void
define void @equal_cost_i32(<4 x i32> %a, <4 x i32> %b) #0 {
entry:
  %a.lo = shufflevector <4 x i32> %a, <4 x i32> poison,
                        <2 x i32> <i32 0, i32 1>
  %b.lo = shufflevector <4 x i32> %b, <4 x i32> poison,
                        <2 x i32> <i32 0, i32 1>
  %lo = add <2 x i32> %a.lo, %b.lo
  %a.hi = shufflevector <4 x i32> %a, <4 x i32> poison,
                        <2 x i32> <i32 2, i32 3>
  %b.hi = shufflevector <4 x i32> %b, <4 x i32> poison,
                        <2 x i32> <i32 2, i32 3>
  %hi = add <2 x i32> %a.hi, %b.hi
  call void @use_narrow_i32(<2 x i32> %lo, <2 x i32> %hi)
  ret void
}

; Keep slice-path discovery linear when a deep no-slice DAG shares both arms at
; every level.

; DEFAULT-LABEL: define <2 x i32> @deep_shared_no_slice(
; DEFAULT-COUNT-18: select <2 x i1>
; DEFAULT:       ret <2 x i32>
define <2 x i32> @deep_shared_no_slice(<2 x i1> %cond, <2 x i32> %x) #0 {
entry:
  %s0 = select <2 x i1> %cond, <2 x i32> %x, <2 x i32> %x
  %s1 = select <2 x i1> %cond, <2 x i32> %s0, <2 x i32> %s0
  %s2 = select <2 x i1> %cond, <2 x i32> %s1, <2 x i32> %s1
  %s3 = select <2 x i1> %cond, <2 x i32> %s2, <2 x i32> %s2
  %s4 = select <2 x i1> %cond, <2 x i32> %s3, <2 x i32> %s3
  %s5 = select <2 x i1> %cond, <2 x i32> %s4, <2 x i32> %s4
  %s6 = select <2 x i1> %cond, <2 x i32> %s5, <2 x i32> %s5
  %s7 = select <2 x i1> %cond, <2 x i32> %s6, <2 x i32> %s6
  %s8 = select <2 x i1> %cond, <2 x i32> %s7, <2 x i32> %s7
  %s9 = select <2 x i1> %cond, <2 x i32> %s8, <2 x i32> %s8
  %s10 = select <2 x i1> %cond, <2 x i32> %s9, <2 x i32> %s9
  %s11 = select <2 x i1> %cond, <2 x i32> %s10, <2 x i32> %s10
  %s12 = select <2 x i1> %cond, <2 x i32> %s11, <2 x i32> %s11
  %s13 = select <2 x i1> %cond, <2 x i32> %s12, <2 x i32> %s12
  %s14 = select <2 x i1> %cond, <2 x i32> %s13, <2 x i32> %s13
  %s15 = select <2 x i1> %cond, <2 x i32> %s14, <2 x i32> %s14
  %s16 = select <2 x i1> %cond, <2 x i32> %s15, <2 x i32> %s15
  %s17 = select <2 x i1> %cond, <2 x i32> %s16, <2 x i32> %s16
  ret <2 x i32> %s17
}

attributes #0 = { "target-cpu"="gfx950" }
attributes #1 = { nounwind willreturn memory(none) }
attributes #2 = { convergent nounwind willreturn memory(none) }
