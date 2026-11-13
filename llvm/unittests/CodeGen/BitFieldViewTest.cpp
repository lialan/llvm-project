//===- BitFieldViewTest.cpp - Bit-field view arithmetic -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Exhaustive checks of the shared arithmetic of the bit-field views on small
// widths: every step and every direct form is compared with the definition of
// the field on all values.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/BitFieldView.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/KnownBits.h"
#include "gtest/gtest.h"

using namespace llvm;

/// Ext(S[F.Lo, F.Lo + F.Width)) as a value of \p Bits bits.
static APInt evalField(const BitField &F, const APInt &S, unsigned Bits) {
  APInt Field = S.extractBits(F.Width, F.Lo);
  switch (F.Ext) {
  case BitFieldExtKind::None:
    EXPECT_EQ(F.Width, Bits);
    return Field;
  case BitFieldExtKind::Zero:
    return Field.zext(Bits);
  case BitFieldExtKind::Sign:
    return Field.sext(Bits);
  }
  llvm_unreachable("unknown bit-field extension");
}

/// Applies \p Form to \p V.
static APInt evalForm(ArrayRef<BitFieldFormOp> Form, APInt V) {
  for (const BitFieldFormOp &Op : Form) {
    const unsigned Bits = V.getBitWidth();
    switch (Op.Kind) {
    case BitFieldFormOp::Trunc:
      V = V.trunc(Op.Bits);
      break;
    case BitFieldFormOp::ZExt:
      V = V.zext(Op.Bits);
      break;
    case BitFieldFormOp::SExt:
      V = V.sext(Op.Bits);
      break;
    case BitFieldFormOp::LShr:
      V = V.lshr(Op.Imm0);
      break;
    case BitFieldFormOp::AShr:
      V = V.ashr(Op.Imm0);
      break;
    case BitFieldFormOp::AndLow:
      V &= APInt::getLowBitsSet(Bits, Op.Imm0);
      break;
    case BitFieldFormOp::SExtInReg:
      V = V.trunc(Op.Imm0).sext(Bits);
      break;
    case BitFieldFormOp::UBFX:
      EXPECT_LE(Op.Imm0 + Op.Imm1, Bits);
      V = V.extractBits(Op.Imm1, Op.Imm0).zext(Bits);
      break;
    case BitFieldFormOp::SBFX:
      EXPECT_LE(Op.Imm0 + Op.Imm1, Bits);
      V = V.extractBits(Op.Imm1, Op.Imm0).sext(Bits);
      break;
    }
    EXPECT_EQ(V.getBitWidth(), Op.Bits);
  }
  return V;
}

/// Calls \p Fn for each field of a value of \p SrcBits bits that describes a
/// value of \p Bits bits.
template <typename FnT>
static void forEachField(unsigned SrcBits, unsigned Bits, FnT Fn) {
  for (unsigned Lo = 0; Lo < SrcBits; ++Lo) {
    for (unsigned Width = 1; Lo + Width <= SrcBits && Width <= Bits; ++Width) {
      if (Width == Bits) {
        Fn(BitField{Lo, Width, BitFieldExtKind::None});
        continue;
      }
      Fn(BitField{Lo, Width, BitFieldExtKind::Zero});
      Fn(BitField{Lo, Width, BitFieldExtKind::Sign});
    }
  }
}

static constexpr unsigned Widths[] = {1, 2, 3, 5, 8};

/// D[I] = S[Off + I] for I < Avail, filled above by Fill (unspecified bits
/// are zeros or ones by \p FillOnes).
static APInt evalThrough(const APInt &S, unsigned Bits, unsigned Off,
                         unsigned Avail, BitFieldExtKind Fill, bool FillOnes) {
  APInt D = S.extractBits(Avail, Off);
  switch (Fill) {
  case BitFieldExtKind::None:
    return FillOnes ? D.zext(Bits) | APInt::getBitsSetFrom(Bits, Avail)
                    : D.zext(Bits);
  case BitFieldExtKind::Zero:
    return D.zext(Bits);
  case BitFieldExtKind::Sign:
    return D.sext(Bits);
  }
  llvm_unreachable("unknown bit-field extension");
}

namespace {

TEST(BitFieldViewTest, FormsComputeTheField) {
  for (unsigned SrcBits : Widths) {
    for (unsigned Bits : Widths) {
      forEachField(SrcBits, Bits, [&](const BitField &F) {
        SmallVector<BitFieldForm, 8> Forms;
        getBitFieldForms(F, Bits, SrcBits, Forms);
        ASSERT_FALSE(Forms.empty());
        for (const BitFieldForm &Form : Forms) {
          for (uint64_t S = 0; S != (uint64_t(1) << SrcBits); ++S) {
            APInt Src(SrcBits, S);
            EXPECT_EQ(evalForm(Form, Src), evalField(F, Src, Bits));
          }
        }
      });
    }
  }
}

TEST(BitFieldViewTest, StepThroughIsTheSameField) {
  // The view describes a value of RootBits bits, wider than every field below
  // unless the field is the whole value.
  const unsigned SrcBits = 6;
  for (unsigned Bits : {1u, 3u, 6u, 8u}) {
    for (unsigned Off = 0; Off < SrcBits; ++Off) {
      for (unsigned Avail = 1; Off + Avail <= SrcBits && Avail <= Bits;
           ++Avail) {
        for (BitFieldExtKind Fill :
             {BitFieldExtKind::None, BitFieldExtKind::Zero,
              BitFieldExtKind::Sign}) {
          for (unsigned RootBits : {Bits, 9u}) {
            forEachField(Bits, RootBits, [&](const BitField &F) {
              std::optional<BitField> G =
                  F.stepThrough(Off, Avail, SrcBits, Fill);
              if (!G)
                return;
              for (uint64_t S = 0; S != (uint64_t(1) << SrcBits); ++S) {
                APInt Src(SrcBits, S);
                APInt Expected = evalField(*G, Src, RootBits);
                for (bool FillOnes : {false, true})
                  EXPECT_EQ(evalField(F,
                                      evalThrough(Src, Bits, Off, Avail, Fill,
                                                  FillOnes),
                                      RootBits),
                            Expected);
              }
            });
          }
        }
      }
    }
  }
}

TEST(BitFieldViewTest, StepThroughKnownFields) {
  // A field that is not all zeros and does not read unspecified bits has a
  // step, unless it is a zero extension of a sign extension.
  BitField F{2, 4, BitFieldExtKind::Zero};
  EXPECT_FALSE(F.stepThrough(0, 4, 8, BitFieldExtKind::None));
  EXPECT_FALSE(F.stepThrough(0, 4, 8, BitFieldExtKind::Sign));
  std::optional<BitField> G = F.stepThrough(0, 4, 8, BitFieldExtKind::Zero);
  ASSERT_TRUE(G);
  EXPECT_EQ(G->Lo, 2u);
  EXPECT_EQ(G->Width, 2u);
  EXPECT_EQ(G->Ext, BitFieldExtKind::Zero);
  // A field of zeros is a constant.
  EXPECT_FALSE(F.stepThrough(0, 2, 8, BitFieldExtKind::Zero));
  // Copies of the sign bit are the extension of that bit.
  G = F.stepThrough(1, 2, 8, BitFieldExtKind::Sign);
  EXPECT_FALSE(G);
  G = BitField{2, 4, BitFieldExtKind::Sign}.stepThrough(1, 2, 8,
                                                        BitFieldExtKind::Sign);
  ASSERT_TRUE(G);
  EXPECT_EQ(G->Lo, 2u);
  EXPECT_EQ(G->Width, 1u);
  EXPECT_EQ(G->Ext, BitFieldExtKind::Sign);
}

TEST(BitFieldViewTest, StepThroughMaskIsTheSameField) {
  const unsigned Bits = 6;
  for (uint64_t X = 0; X != (uint64_t(1) << Bits); ++X) {
    KnownBits Known = KnownBits::makeConstant(APInt(Bits, X));
    for (unsigned RootBits : {Bits, 9u}) {
      forEachField(Bits, RootBits, [&](const BitField &F) {
        for (bool IsAnd : {true, false}) {
          std::optional<BitField> G = F.stepThroughMask(IsAnd, Known);
          if (!G)
            continue;
          for (uint64_t Y = 0; Y != (uint64_t(1) << Bits); ++Y) {
            APInt YV(Bits, Y), XV(Bits, X);
            EXPECT_EQ(evalField(F, IsAnd ? YV & XV : YV | XV, RootBits),
                      evalField(*G, YV, RootBits));
            if (!IsAnd)
              EXPECT_EQ(evalField(F, YV ^ XV, RootBits),
                        evalField(*G, YV, RootBits));
          }
        }
      });
    }
  }
}

TEST(BitFieldViewTest, TwoFieldMaskIsNotAField) {
  // 0xf0f keeps two fields of the low 12 bits.
  KnownBits Known = KnownBits::makeConstant(APInt(12, 0xf0f));
  EXPECT_FALSE(
      (BitField{0, 12, BitFieldExtKind::None}).stepThroughMask(true, Known));
  // A narrower field may select one of them.
  std::optional<BitField> G =
      BitField{8, 4, BitFieldExtKind::Zero}.stepThroughMask(true, Known);
  ASSERT_TRUE(G);
  EXPECT_EQ(G->Lo, 8u);
  EXPECT_EQ(G->Width, 4u);
}

} // end anonymous namespace
