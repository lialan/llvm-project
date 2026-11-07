//===- AMDGPUShuffleUtils.h - Shuffle helpers shared by ISel -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shuffle decisions shared by SelectionDAG and GlobalISel, so that both
// selectors split shuffles into aligned register pairs and select
// V_PK_MOV_B32 under the same rules.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSHUFFLEUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSHUFFLEUTILS_H

#include <cstdint>

namespace llvm::AMDGPU {

/// A shuffle mask element expressed as a lane of an aligned pair of source
/// elements.
struct ShufflePairLane {
  /// Shuffle operand the element is read from, or -1 for an undef element.
  int Src = -1;
  /// Index of the first element of the aligned pair within the operand.
  int PairIdx = 0;
  /// Lane within the pair.
  int Lane = 0;

  bool isUndef() const { return Src < 0; }
};

/// Decompose shuffle mask element \p MaskElt of a shuffle whose operands have
/// \p SrcNumElts elements each.
ShufflePairLane getShufflePairLane(int MaskElt, int SrcNumElts);

/// Return true if a 2-element shuffle result reads the odd lane of one aligned
/// pair into its low lane and the even lane of another into its high lane.
/// This blend cannot be done with subregister copies, but V_PK_MOV_B32 can do
/// it for 32-bit elements.
bool isOddToEvenPairBlend(ShufflePairLane Lo, ShufflePairLane Hi);

/// Return true if the \p SliceBits wide slice starting at element \p Index of
/// a vector with \p EltBits wide elements is a whole aligned 64-bit register
/// pair, so it can be read as a subregister without alignment copies.
bool isAlignedRegPairSlice(unsigned EltBits, unsigned SliceBits,
                           uint64_t Index);

/// Return the source modifiers for a V_PK_MOV_B32 operand that supplies lane
/// \p L.Lane of a 64-bit register pair.
unsigned getPkMovB32SrcMods(ShufflePairLane L);

} // namespace llvm::AMDGPU

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUSHUFFLEUTILS_H
