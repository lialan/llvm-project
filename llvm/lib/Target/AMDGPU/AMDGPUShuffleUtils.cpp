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

static bool isSamePairInOrder(AMDGPU::ShufflePairLane Lo,
                              AMDGPU::ShufflePairLane Hi) {
  return !Lo.isUndef() && Lo.Src == Hi.Src && Lo.PairIdx == Hi.PairIdx &&
         Lo.Lane == 0 && Hi.Lane == 1;
}

SmallVector<AMDGPU::ShufflePiece, 16>
AMDGPU::planShuffleByPairs(ArrayRef<int> Mask, int SrcNumElts, unsigned EltBits,
                           bool HasPkMovB32) {
  assert(Mask.size() % 2 == 0 && "result must split into pairs");

  // We're trying to form pieces that the register allocation pipeline won't
  // be able to figure out, like how to use v_pk_mov_b32 to do a register
  // blend or 16-bit op_sel. It should be able to figure out how to reassemble
  // a pair of copies into a consecutive register copy, so use individual
  // elements unless a subvector or a 2-element shuffle helps.
  //
  // TODO: This is a bit of hack, and we should probably always use the largest
  // possible subvector we can (or at least use it for aligned pairs). However
  // we have worse support for combines on them, and the DAG scheduler also
  // ends up doing a worse job with the extract_subvectors.
  SmallVector<ShufflePiece, 16> Pieces;
  for (size_t I = 0, E = Mask.size(); I != E; I += 2) {
    ShufflePairLane Lo = getShufflePairLane(Mask[I], SrcNumElts);
    ShufflePairLane Hi = getShufflePairLane(Mask[I + 1], SrcNumElts);
    ShufflePieceKind Kind = ShufflePieceKind::Elements;
    if (EltBits == 16 && isSamePairInOrder(Lo, Hi))
      Kind = ShufflePieceKind::PairSlice;
    else if (EltBits == 32 && HasPkMovB32 && isOddToEvenPairBlend(Lo, Hi))
      Kind = ShufflePieceKind::PairBlend;
    Pieces.push_back({Kind, Lo, Hi});
  }
  return Pieces;
}

bool AMDGPU::shouldLowerShuffleByPairs(unsigned EltBits, unsigned DstNumElts,
                                       unsigned SrcNumElts,
                                       bool HasVOP3PInsts) {
  if (DstNumElts % 2 != 0 || SrcNumElts % 2 != 0 || DstNumElts > 32 ||
      SrcNumElts > 32)
    return false;
  if (EltBits == 32)
    return true;
  return EltBits == 16 && HasVOP3PInsts && DstNumElts >= 4;
}
