//===- AMDGPUShuffleUtils.cpp - Shuffle helpers shared by ISel -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUShuffleUtils.h"
#include "SIDefines.h"
#include <cassert>

using namespace llvm;

AMDGPU::ShufflePairLane AMDGPU::getShufflePairLane(int MaskElt,
                                                   int SrcNumElts) {
  assert(SrcNumElts % 2 == 0 && "operands must split into pairs");
  if (MaskElt < 0)
    return {};

  int Src = MaskElt < SrcNumElts ? 0 : 1;
  int Elt = MaskElt - Src * SrcNumElts;
  return {Src, Elt & ~1, Elt & 1};
}

bool AMDGPU::isOddToEvenPairBlend(ShufflePairLane Lo, ShufflePairLane Hi) {
  return !Lo.isUndef() && !Hi.isUndef() && Lo.Lane == 1 && Hi.Lane == 0;
}

bool AMDGPU::isAlignedRegPairSlice(unsigned EltBits, unsigned SliceBits,
                                   uint64_t Index) {
  // Packed 16-bit slices stay out: narrowing shuffles and reductions can
  // introduce unused-lane operations and increase register pressure.
  return EltBits == 32 && SliceBits == 64 && (Index * EltBits) % SliceBits == 0;
}

unsigned AMDGPU::getPkMovB32SrcMods(ShufflePairLane L) {
  // op_sel selects the high half of the operand. Also set op_sel_hi to avoid
  // printing it; it has no effect on the result.
  return (L.Lane ? SISrcMods::OP_SEL_0 : SISrcMods::NONE) | SISrcMods::OP_SEL_1;
}
