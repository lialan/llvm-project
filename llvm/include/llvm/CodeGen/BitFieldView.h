//===- llvm/CodeGen/BitFieldView.h - Bit-field views of values --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// A bit-field view describes an integer value C of W bits as the field
/// S[Lo, Lo + Width) of a source value S, zero- or sign-extended to W bits,
/// or not extended at all if Width == W. A view is found by walking back from
/// C through operations that only select, move or extend bits, and nested
/// steps compose: (((X >> 4) & 255) >> 2) & 15 is the zero extension of
/// X[6, 10).
///
/// This file holds what the GlobalISel walker (forEachBitFieldView in
/// GlobalISel/Utils.h) and the SelectionDAG walker
/// (SelectionDAG::forEachBitFieldView) share: the arithmetic of one step and
/// the direct forms of a view.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_BITFIELDVIEW_H
#define LLVM_CODEGEN_BITFIELDVIEW_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include <cstdint>
#include <optional>

namespace llvm {

struct KnownBits;

/// How a bit field is extended to the width of the value that it describes.
enum class BitFieldExtKind : uint8_t {
  None, ///< The field is the whole value.
  Zero,
  Sign,
};

/// The field [Lo, Lo + Width) of a value, and how it is extended to the width
/// W of the value that a view describes. Ext is None iff Width == W.
struct BitField {
  unsigned Lo = 0;
  unsigned Width = 0;
  BitFieldExtKind Ext = BitFieldExtKind::None;

  /// Steps back through an operation whose result D has the bits
  /// D[I] = S[Off + I] for I < \p Avail, where S is an operand of \p SrcBits
  /// bits. The bits D[I] for I >= Avail are zeros (\p Fill Zero), copies of
  /// S[Off + Avail - 1] (\p Fill Sign), or unspecified (\p Fill None).
  /// Returns the field of S that this field of D is, or std::nullopt if there
  /// is none: the field covers unspecified bits, is all zeros, or would be a
  /// zero extension of a sign extension.
  LLVM_ABI std::optional<BitField> stepThrough(unsigned Off, unsigned Avail,
                                               unsigned SrcBits,
                                               BitFieldExtKind Fill) const;

  /// Steps back through (and Y, X), (or Y, X) or (xor Y, X), where the known
  /// bits \p KnownX of X select this field of Y: for an AND, X is one in the
  /// low bits of the field and zero above them; for an OR or an XOR, X is
  /// zero in the whole field. Returns the field of Y, or std::nullopt.
  LLVM_ABI std::optional<BitField>
  stepThroughMask(bool IsAnd, const KnownBits &KnownX) const;
};

/// The view of a value C as Ext(Src[Lo, Lo + Width)).
template <typename ValueT> struct BitFieldView : BitField {
  ValueT Src;

  BitFieldView(const BitField &F, ValueT Src) : BitField(F), Src(Src) {}
};

/// One operation of a direct form of a bit-field view. A form applies its
/// operations in order, the first one to the source of the view; Bits is the
/// width of the result of the operation.
struct BitFieldFormOp {
  enum OpKind : uint8_t {
    Trunc,     ///< Truncate to Bits.
    ZExt,      ///< Zero-extend to Bits.
    SExt,      ///< Sign-extend to Bits.
    LShr,      ///< Logical shift right by Imm0.
    AShr,      ///< Arithmetic shift right by Imm0.
    AndLow,    ///< Keep the low Imm0 bits, clear the others.
    SExtInReg, ///< Sign-extend the low Imm0 bits.
    UBFX,      ///< Zero-extend the Imm1 bits that start at bit Imm0.
    SBFX,      ///< Sign-extend the Imm1 bits that start at bit Imm0.
  };

  OpKind Kind;
  unsigned Bits;
  unsigned Imm0 = 0;
  unsigned Imm1 = 0;
};

/// A direct form of a view. An empty form is the source itself.
using BitFieldForm = SmallVector<BitFieldFormOp, 3>;

/// Appends to \p Forms the direct forms of the field \p F of a value of
/// \p SrcBits bits, as a value of \p Bits bits, the most direct first.
LLVM_ABI void getBitFieldForms(const BitField &F, unsigned Bits,
                               unsigned SrcBits,
                               SmallVectorImpl<BitFieldForm> &Forms);

} // end namespace llvm

#endif // LLVM_CODEGEN_BITFIELDVIEW_H
