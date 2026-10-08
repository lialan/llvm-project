//===- RegisterClassInfoTest.cpp - Register cost cache tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/RegisterClassInfo.h"
#include "RISCVRegisterInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"
#include <memory>

using namespace llvm;

namespace {

class RegisterClassInfoTest : public testing::TestWithParam<const char *> {
protected:
  LLVMContext Context;
  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<Module> M;
  std::unique_ptr<MachineModuleInfo> MMI;

  static void SetUpTestSuite() {
    LLVMInitializeRISCVTargetInfo();
    LLVMInitializeRISCVTarget();
    LLVMInitializeRISCVTargetMC();
  }

  void SetUp() override {
    Triple TT(GetParam());
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget(TT, Error);
    ASSERT_NE(T, nullptr) << Error;
    TM.reset(T->createTargetMachine(TT, "generic", "+c", TargetOptions(),
                                    std::nullopt));
    ASSERT_TRUE(TM);
    M = std::make_unique<Module>("register-costs", Context);
    M->setDataLayout(TT.computeDataLayout());
    MMI = std::make_unique<MachineModuleInfo>(TM.get());
  }

  MachineFunction &createFunction(StringRef Name, bool OptSize) {
    auto *FT = FunctionType::get(Type::getVoidTy(Context), false);
    auto *F = Function::Create(FT, GlobalValue::ExternalLinkage, Name, *M);
    if (OptSize)
      F->addFnAttr(Attribute::OptimizeForSize);
    MachineFunction &MF = MMI->getOrCreateMachineFunction(*F);
    MF.getRegInfo().freezeReservedRegs();
    return MF;
  }
};

TEST_P(RegisterClassInfoTest, CostTableChangesInvalidateCachedClassInfo) {
  // RISC-V with C charges for non-compressible registers only under optsize.
  // The functions otherwise share the inputs that invalidate the class cache.
  MachineFunction &Normal = createFunction("normal", false);
  MachineFunction &OptSize = createFunction("optsize", true);
  ASSERT_EQ(Normal.getSubtarget().getRegisterInfo(),
            OptSize.getSubtarget().getRegisterInfo());
  ASSERT_EQ(Normal.getRegInfo().getReservedRegs(),
            OptSize.getRegInfo().getReservedRegs());
  ASSERT_EQ(Normal.getRegInfo().getCalleeSavedRegs(),
            OptSize.getRegInfo().getCalleeSavedRegs());

  // Exercise both transition directions, including returning to the first
  // table.
  for (MachineFunction *First : {&Normal, &OptSize}) {
    MachineFunction *Second = First == &Normal ? &OptSize : &Normal;
    RegisterClassInfo Cached;
    for (MachineFunction *MF : {First, Second, First}) {
      SCOPED_TRACE(MF->getName());
      Cached.runOnMachineFunction(*MF);
      RegisterClassInfo Fresh;
      Fresh.runOnMachineFunction(*MF);

      // X5 cannot use the compact register encoding, while GPR contains both
      // compressible and non-compressible registers.
      EXPECT_EQ(Fresh.getMinCost(&RISCV::GPRX5RegClass),
                MF == &OptSize ? 1 : 0);
      if (MF == &OptSize)
        EXPECT_GT(Fresh.getLastCostChange(&RISCV::GPRRegClass), 0u);
      else
        EXPECT_EQ(Fresh.getLastCostChange(&RISCV::GPRRegClass), 0u);
      EXPECT_EQ(Cached.getMinCost(&RISCV::GPRX5RegClass),
                Fresh.getMinCost(&RISCV::GPRX5RegClass));
      EXPECT_EQ(Cached.getLastCostChange(&RISCV::GPRRegClass),
                Fresh.getLastCostChange(&RISCV::GPRRegClass));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(RISCV, RegisterClassInfoTest,
                         testing::Values("riscv32-unknown-linux",
                                         "riscv64-unknown-linux"));

} // namespace
