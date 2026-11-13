//===- BitFieldView.cpp - Bit-field views of values -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/BitFieldView.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"

using namespace llvm;

/// The extension of a view Outer(F) after the field F has been rewritten as
/// Fill(F') with a field F' of fewer bits: Outer(Fill(F')) as an extension of
/// F', or std::nullopt if it is none.
static std::optional<BitFieldExtKind> composeExt(BitFieldExtKind Outer,
                                                 BitFieldExtKind Fill) {
  switch (Fill) {
  case BitFieldExtKind::None:
    return Outer;
  case BitFieldExtKind::Zero:
    // The top bit of a zero extension of F' is zero, so a sign extension of it
    // is a zero extension too.
    return BitFieldExtKind::Zero;
  case BitFieldExtKind::Sign:
    // A zero extension of a sign extension of F' is neither.
    if (Outer == BitFieldExtKind::Zero)
      return std::nullopt;
    return BitFieldExtKind::Sign;
  }
  llvm_unreachable("unknown bit-field extension");
}

std::optional<BitField> BitField::stepThrough(unsigned Off, unsigned Avail,
                                              unsigned SrcBits,
                                              BitFieldExtKind Fill) const {
  if (Avail == 0 || uint64_t(Off) + Avail > SrcBits)
    return std::nullopt;
  BitField S;
  BitFieldExtKind SrcFill = BitFieldExtKind::None;
  if (Lo + Width <= Avail) {
    S.Lo = Off + Lo;
    S.Width = Width;
  } else if (Fill == BitFieldExtKind::None) {
    // The field covers bits that the operation leaves unspecified.
    return std::nullopt;
  } else if (Lo < Avail) {
    S.Lo = Off + Lo;
    S.Width = Avail - Lo;
    SrcFill = Fill;
  } else if (Fill == BitFieldExtKind::Zero) {
    // A field of zeros is a constant, not a field.
    return std::nullopt;
  } else {
    // A field of copies of the sign bit is the extension of that bit.
    S.Lo = Off + Avail - 1;
    S.Width = 1;
    if (Width != 1)
      SrcFill = BitFieldExtKind::Sign;
  }
  std::optional<BitFieldExtKind> E = composeExt(Ext, SrcFill);
  if (!E)
    return std::nullopt;
  S.Ext = *E;
  return S;
}

std::optional<BitField>
BitField::stepThroughMask(bool IsAnd, const KnownBits &KnownX) const {
  if (KnownX.getBitWidth() < Lo + Width)
    return std::nullopt;
  APInt Zero = KnownX.Zero.extractBits(Width, Lo);
  unsigned Keep = Width;
  if (IsAnd) {
    // Low ones, then zeros: the AND keeps the low Keep bits of the field.
    Keep = KnownX.One.extractBits(Width, Lo).countr_one();
    if (Keep == 0 || !(Zero | APInt::getLowBitsSet(Width, Keep)).isAllOnes())
      return std::nullopt;
  } else if (!Zero.isAllOnes()) {
    return std::nullopt;
  }
  if (Keep == Width)
    return *this;
  return BitField{Lo, Keep, BitFieldExtKind::Zero};
}

/// Appends the forms of the field \p F of a value B, where B is \p Prefix
/// applied to the source and has \p Bits bits. The bits of B from \p Top up are
/// zeros (F.Ext Zero) or copies of bit Top - 1 (F.Ext Sign); Top is Bits if
/// there are none.
static void addSameWidthForms(const BitFieldForm &Prefix, unsigned Bits,
                              unsigned Top, const BitField &F,
                              SmallVectorImpl<BitFieldForm> &Forms) {
  using Op = BitFieldFormOp;
  auto Add = [&](std::initializer_list<BitFieldFormOp> Ops) {
    BitFieldForm Form(Prefix);
    Form.append(Ops.begin(), Ops.end());
    Forms.push_back(std::move(Form));
  };
  const unsigned Lo = F.Lo, Width = F.Width;
  if (F.Ext == BitFieldExtKind::None || (Lo == 0 && Width == Top)) {
    Add({});
    return;
  }
  if (F.Ext == BitFieldExtKind::Zero) {
    if (Lo + Width == Top) {
      Add({{Op::LShr, Bits, Lo}});
    } else if (Lo == 0) {
      Add({{Op::AndLow, Bits, Width}});
    } else {
      Add({{Op::UBFX, Bits, Lo, Width}});
      Add({{Op::LShr, Bits, Lo}, {Op::AndLow, Bits, Width}});
    }
    return;
  }
  if (Lo + Width == Top) {
    Add({{Op::AShr, Bits, Lo}});
  } else if (Lo == 0) {
    Add({{Op::SExtInReg, Bits, Width}});
    Add({{Op::Shl, Bits, Bits - Width}, {Op::AShr, Bits, Bits - Width}});
  } else {
    Add({{Op::SBFX, Bits, Lo, Width}});
    Add({{Op::LShr, Bits, Lo}, {Op::SExtInReg, Bits, Width}});
    Add({{Op::Shl, Bits, Bits - Lo - Width}, {Op::AShr, Bits, Bits - Width}});
  }
}

void llvm::getBitFieldForms(const BitField &F, unsigned Bits, unsigned SrcBits,
                            SmallVectorImpl<BitFieldForm> &Forms) {
  using Op = BitFieldFormOp;
  assert(F.Width >= 1 && F.Lo + F.Width <= SrcBits && "field out of range");
  assert((F.Ext == BitFieldExtKind::None) == (F.Width == Bits) &&
         "only a narrower field is extended");
  if (Bits == SrcBits) {
    addSameWidthForms({}, Bits, Bits, F, Forms);
    return;
  }

  if (Bits < SrcBits) {
    // The field is inside the low Bits bits of the source: truncate first.
    if (F.Lo + F.Width <= Bits) {
      addSameWidthForms({{Op::Trunc, Bits}}, Bits, Bits, F, Forms);
      return;
    }
    // Otherwise shift the field down in the type of the source first.
    if (F.Ext == BitFieldExtKind::None || F.Lo + F.Width == SrcBits) {
      Op::OpKind Shift = F.Ext == BitFieldExtKind::Sign ? Op::AShr : Op::LShr;
      Forms.push_back({{Shift, SrcBits, F.Lo}, {Op::Trunc, Bits}});
      return;
    }
    if (F.Ext == BitFieldExtKind::Zero) {
      Forms.push_back({{Op::LShr, SrcBits, F.Lo},
                       {Op::Trunc, Bits},
                       {Op::AndLow, Bits, F.Width}});
      Forms.push_back({{Op::UBFX, SrcBits, F.Lo, F.Width}, {Op::Trunc, Bits}});
      return;
    }
    Forms.push_back({{Op::LShr, SrcBits, F.Lo},
                     {Op::Trunc, Bits},
                     {Op::SExtInReg, Bits, F.Width}});
    Forms.push_back({{Op::SBFX, SrcBits, F.Lo, F.Width}, {Op::Trunc, Bits}});
    return;
  }

  // The field of a narrower source is extended: take the field in the type of
  // the source and extend it, or extend the source and take the field.
  const Op::OpKind ExtOp = F.Ext == BitFieldExtKind::Sign ? Op::SExt : Op::ZExt;
  BitField Narrow = F;
  if (F.Width == SrcBits)
    Narrow.Ext = BitFieldExtKind::None;
  SmallVector<BitFieldForm, 3> NarrowForms;
  addSameWidthForms({}, SrcBits, SrcBits, Narrow, NarrowForms);
  for (BitFieldForm &Form : NarrowForms) {
    Form.push_back({ExtOp, Bits});
    Forms.push_back(std::move(Form));
  }
  addSameWidthForms({{ExtOp, Bits}}, Bits, SrcBits, F, Forms);
  // A form that takes a field below the top of the source and extends it by
  // itself does not read the bits that an any-extension adds. With Top ==
  // Bits, addSameWidthForms only gives such forms.
  if (F.Width != SrcBits)
    addSameWidthForms({{Op::AnyExt, Bits}}, Bits, Bits, F, Forms);
}
