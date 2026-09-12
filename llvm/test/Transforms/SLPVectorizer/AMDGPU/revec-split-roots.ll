; REQUIRES: amdgpu-registered-target
; RUN: opt -S -passes=slp-vectorizer < %s | FileCheck %s \
; RUN:   --check-prefix=DEFAULT
; RUN: opt -S -passes=slp-vectorizer -slp-revec < %s | FileCheck %s \
; RUN:   --check-prefix=REVEC

target triple = "amdgcn-amd-amdhsa"

; Scalar lanes extracted from existing vectors are rebuilt into independent v2
; roots, so default SLP does not see one v4 root. ReVec is a control showing
; that the complete source-derived tree can be recovered.
;
; DEFAULT-LABEL: define void @split_buildvector_roots(
; DEFAULT-NOT:     select <4 x i1>
; DEFAULT:         select <2 x i1>
; DEFAULT-NOT:     select <4 x i1>
; DEFAULT:         select <2 x i1>
; DEFAULT-NOT:     select <4 x i1>
; DEFAULT:         ret void
;
; REVEC-LABEL: define void @split_buildvector_roots(
; REVEC:         [[SELECT:%.*]] = select <4 x i1> %cond, <4 x float> %a,
; REVEC-SAME:      <4 x float> zeroinitializer
; REVEC-NOT:     select <2 x i1>
; REVEC:         ret void
define void @split_buildvector_roots(<4 x float> %a, <4 x i1> %cond,
                                     ptr addrspace(3) %out0,
                                     ptr addrspace(3) %out1) #0 {
entry:
  %a0 = extractelement <4 x float> %a, i32 0
  %a1 = extractelement <4 x float> %a, i32 1
  %a2 = extractelement <4 x float> %a, i32 2
  %a3 = extractelement <4 x float> %a, i32 3
  %p0 = extractelement <4 x i1> %cond, i32 0
  %p1 = extractelement <4 x i1> %cond, i32 1
  %p2 = extractelement <4 x i1> %cond, i32 2
  %p3 = extractelement <4 x i1> %cond, i32 3
  %s0 = select i1 %p0, float %a0, float 0.0
  %s1 = select i1 %p1, float %a1, float 0.0
  %s2 = select i1 %p2, float %a2, float 0.0
  %s3 = select i1 %p3, float %a3, float 0.0
  %v01.0 = insertelement <2 x float> poison, float %s0, i32 0
  %v01 = insertelement <2 x float> %v01.0, float %s1, i32 1
  %v23.0 = insertelement <2 x float> poison, float %s2, i32 0
  %v23 = insertelement <2 x float> %v23.0, float %s3, i32 1
  store <2 x float> %v01, ptr addrspace(3) %out0, align 8
  store <2 x float> %v23, ptr addrspace(3) %out1, align 8
  ret void
}

attributes #0 = { "target-cpu"="gfx950" }
