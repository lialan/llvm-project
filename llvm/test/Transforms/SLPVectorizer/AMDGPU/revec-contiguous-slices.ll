; REQUIRES: amdgpu-registered-target
; RUN: opt -S -passes=slp-vectorizer -slp-revec < %s | FileCheck %s
; RUN: opt -S -passes=slp-vectorizer -slp-revec -slp-max-vf=4 \
; RUN:   -slp-max-reg-size=256 -slp-threshold=-100 < %s | FileCheck %s \
; RUN:   --check-prefix=BARRIER
; RUN: opt -S -passes=slp-vectorizer -slp-revec -slp-max-vf=2 < %s | \
; RUN:   FileCheck %s --check-prefix=LIMIT
; RUN: opt -S -passes=slp-vectorizer -slp-revec -slp-max-reg-size=128 < %s | \
; RUN:   FileCheck %s --check-prefix=LIMIT

target triple = "amdgcn-amd-amdhsa"

; A complete group of vector operations on contiguous source slices can reuse
; the source vectors instead of gathering repeated copies of them.

; CHECK-LABEL: define void @rootless_slices(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ADD:%.*]] = fadd <8 x float> %a, %b
; CHECK-NEXT:    [[SELECT:%.*]] = select <8 x i1> %cond, <8 x float> [[ADD]], <8 x float> zeroinitializer
; CHECK-NEXT:    [[SLICE0:%.*]] = shufflevector <8 x float> [[SELECT]], <8 x float> poison, <2 x i32> <i32 0, i32 1>
; CHECK-NEXT:    store <2 x float> [[SLICE0]], ptr addrspace(3) %out0, align 8
; CHECK-NEXT:    [[SLICE1:%.*]] = shufflevector <8 x float> [[SELECT]], <8 x float> poison, <2 x i32> <i32 2, i32 3>
; CHECK-NEXT:    store <2 x float> [[SLICE1]], ptr addrspace(3) %out1, align 8
; CHECK-NEXT:    [[SLICE2:%.*]] = shufflevector <8 x float> [[SELECT]], <8 x float> poison, <2 x i32> <i32 4, i32 5>
; CHECK-NEXT:    store <2 x float> [[SLICE2]], ptr addrspace(3) %out2, align 8
; CHECK-NEXT:    [[SLICE3:%.*]] = shufflevector <8 x float> [[SELECT]], <8 x float> poison, <2 x i32> <i32 6, i32 7>
; CHECK-NEXT:    store <2 x float> [[SLICE3]], ptr addrspace(3) %out3, align 8
; CHECK-NEXT:    ret void
; BARRIER-LABEL: define void @rootless_slices(
; BARRIER:         fadd <8 x float>
; BARRIER:         ret void
; LIMIT-LABEL: define void @rootless_slices(
; LIMIT-NOT:     fadd <8 x float>
; LIMIT:         ret void
define void @rootless_slices(
    <8 x float> %a, <8 x float> %b, <8 x i1> %cond,
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

; This is the issue shape before the first SLP vectorization: two separate
; stores expose pairs of v2 results, but no common concatenating root covers
; all eight lanes.

; CHECK-LABEL: define void @from_scalar_slp(
; CHECK:         [[ADD:%.*]] = fadd <8 x float> %a, %b
; CHECK-NEXT:    [[SELECT:%.*]] = select <8 x i1> %cond, <8 x float> [[ADD]], <8 x float> zeroinitializer
; CHECK-NEXT:    [[TRUNC:%.*]] = fptrunc <8 x float> [[SELECT]] to <8 x bfloat>
; CHECK-NOT:     fadd <2 x float>
; CHECK:         ret void
define void @from_scalar_slp(<8 x float> %a, <8 x float> %b,
                             <8 x i1> %cond,
                             ptr addrspace(3) %out0,
                             ptr addrspace(3) %out1) #0 {
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
  %p0 = extractelement <8 x i1> %cond, i32 0
  %p1 = extractelement <8 x i1> %cond, i32 1
  %p2 = extractelement <8 x i1> %cond, i32 2
  %p3 = extractelement <8 x i1> %cond, i32 3
  %p4 = extractelement <8 x i1> %cond, i32 4
  %p5 = extractelement <8 x i1> %cond, i32 5
  %p6 = extractelement <8 x i1> %cond, i32 6
  %p7 = extractelement <8 x i1> %cond, i32 7
  %add0 = fadd float %a0, %b0
  %add1 = fadd float %a1, %b1
  %add2 = fadd float %a2, %b2
  %add3 = fadd float %a3, %b3
  %add4 = fadd float %a4, %b4
  %add5 = fadd float %a5, %b5
  %add6 = fadd float %a6, %b6
  %add7 = fadd float %a7, %b7
  %select0 = select i1 %p0, float %add0, float 0.0
  %select1 = select i1 %p1, float %add1, float 0.0
  %select2 = select i1 %p2, float %add2, float 0.0
  %select3 = select i1 %p3, float %add3, float 0.0
  %select4 = select i1 %p4, float %add4, float 0.0
  %select5 = select i1 %p5, float %add5, float 0.0
  %select6 = select i1 %p6, float %add6, float 0.0
  %select7 = select i1 %p7, float %add7, float 0.0
  %pair01.0 = insertelement <2 x float> poison, float %select0, i32 0
  %pair01 = insertelement <2 x float> %pair01.0, float %select1, i32 1
  %bf01 = fptrunc <2 x float> %pair01 to <2 x bfloat>
  %pair23.0 = insertelement <2 x float> poison, float %select2, i32 0
  %pair23 = insertelement <2 x float> %pair23.0, float %select3, i32 1
  %bf23 = fptrunc <2 x float> %pair23 to <2 x bfloat>
  %pair45.0 = insertelement <2 x float> poison, float %select4, i32 0
  %pair45 = insertelement <2 x float> %pair45.0, float %select5, i32 1
  %bf45 = fptrunc <2 x float> %pair45 to <2 x bfloat>
  %pair67.0 = insertelement <2 x float> poison, float %select6, i32 0
  %pair67 = insertelement <2 x float> %pair67.0, float %select7, i32 1
  %bf67 = fptrunc <2 x float> %pair67 to <2 x bfloat>
  %lo = shufflevector <2 x bfloat> %bf01, <2 x bfloat> %bf23,
                      <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x bfloat> %lo, ptr addrspace(3) %out0, align 8
  %hi = shufflevector <2 x bfloat> %bf45, <2 x bfloat> %bf67,
                      <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x bfloat> %hi, ptr addrspace(3) %out1, align 8
  ret void
}

; An incomplete partition does not justify exceeding the target's normal VF.

; CHECK-LABEL: define void @incomplete_slices(
; CHECK:         fadd <2 x float>
; CHECK:         fadd <2 x float>
; CHECK:         fadd <2 x float>
; CHECK-NOT:     fadd <8 x float>
; CHECK:         ret void
define void @incomplete_slices(
    <8 x float> %a, <8 x float> %b, ptr addrspace(3) %out0,
    ptr addrspace(3) %out1, ptr addrspace(3) %out2) #0 {
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
  %a67 = shufflevector <8 x float> %a, <8 x float> poison,
                       <2 x i32> <i32 6, i32 7>
  %b67 = shufflevector <8 x float> %b, <8 x float> poison,
                       <2 x i32> <i32 6, i32 7>
  %add67 = fadd <2 x float> %a67, %b67
  store <2 x float> %add01, ptr addrspace(3) %out0, align 8
  store <2 x float> %add23, ptr addrspace(3) %out1, align 8
  store <2 x float> %add67, ptr addrspace(3) %out2, align 8
  ret void
}

; A complete slice partition on only one operand is not enough to bypass the
; target VF limit. The other non-constant leaf would have to be gathered.

; CHECK-LABEL: define void @mixed_source_and_splat(
; CHECK-NOT:     fadd <8 x float>
; CHECK:         ret void
define void @mixed_source_and_splat(
    <8 x float> %a, <2 x float> %b, ptr addrspace(3) %out0,
    ptr addrspace(3) %out1, ptr addrspace(3) %out2,
    ptr addrspace(3) %out3) #0 {
entry:
  %a01 = shufflevector <8 x float> %a, <8 x float> poison,
                       <2 x i32> <i32 0, i32 1>
  %add01 = fadd <2 x float> %a01, %b
  %a23 = shufflevector <8 x float> %a, <8 x float> poison,
                       <2 x i32> <i32 2, i32 3>
  %add23 = fadd <2 x float> %a23, %b
  %a45 = shufflevector <8 x float> %a, <8 x float> poison,
                       <2 x i32> <i32 4, i32 5>
  %add45 = fadd <2 x float> %a45, %b
  %a67 = shufflevector <8 x float> %a, <8 x float> poison,
                       <2 x i32> <i32 6, i32 7>
  %add67 = fadd <2 x float> %a67, %b
  store <2 x float> %add01, ptr addrspace(3) %out0, align 8
  store <2 x float> %add23, ptr addrspace(3) %out1, align 8
  store <2 x float> %add45, ptr addrspace(3) %out2, align 8
  store <2 x float> %add67, ptr addrspace(3) %out3, align 8
  ret void
}

; Do not revectorize across inline assembly. It may intentionally serve as a
; scheduling barrier even when it has no side effects.

; CHECK-LABEL: define void @inline_asm_barrier(
; CHECK:         [[LO:%.*]] = fadd <2 x float>
; CHECK-NEXT:    [[BAR:%.*]] = call <2 x i32> asm "", "=v,0"(<2 x i32> %token)
; CHECK:         [[HI:%.*]] = fadd <2 x float>
; CHECK-NOT:     fadd <4 x float>
; CHECK-NOT:     fptrunc <4 x float>
; CHECK:         store <2 x i32> [[BAR]], ptr addrspace(3) %bar.out
; BARRIER-LABEL: define void @inline_asm_barrier(
; BARRIER:         [[LO:%.*]] = fadd <2 x float>
; BARRIER-NEXT:    [[BAR:%.*]] = call <2 x i32> asm "", "=v,0"(<2 x i32> %token)
; BARRIER:         [[HI:%.*]] = fadd <2 x float>
; BARRIER-NOT:     fadd <4 x float>
; BARRIER-NOT:     fptrunc <4 x float>
; BARRIER:         store <2 x i32> [[BAR]], ptr addrspace(3) %bar.out
define void @inline_asm_barrier(
    <4 x float> %a, <4 x float> %b, <2 x i32> %token,
    ptr addrspace(3) %bar.out, ptr addrspace(3) %out0,
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
  %lo.trunc = fptrunc <2 x float> %lo to <2 x bfloat>
  %hi.trunc = fptrunc <2 x float> %hi to <2 x bfloat>
  store <2 x i32> %bar, ptr addrspace(3) %bar.out, align 8
  store <2 x bfloat> %lo.trunc, ptr addrspace(3) %out0, align 4
  store <2 x bfloat> %hi.trunc, ptr addrspace(3) %out1, align 4
  ret void
}

attributes #0 = { "target-cpu"="gfx950" }
attributes #1 = { nounwind willreturn memory(none) }
