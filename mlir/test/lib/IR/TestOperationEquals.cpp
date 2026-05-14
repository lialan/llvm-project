//===- TestOperationEquals.cpp - Passes to test OperationEquivalence ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// This pass illustrates the IR def-use chains through printing.
struct TestOperationEqualPass
    : public PassWrapper<TestOperationEqualPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestOperationEqualPass)

  StringRef getArgument() const final { return "test-operations-equality"; }
  StringRef getDescription() const final { return "Test operations equality."; }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Expects two operations at the top-level:
    int opCount = module.getBody()->getOperations().size();
    if (module->hasAttr("test.includes_setup")) {
      if (opCount < 2) {
        module.emitError()
            << "expected at least 2 top-level ops in the module, got "
            << opCount;
        return signalPassFailure();
      }
    } else if (opCount != 2) {
      module.emitError() << "expected 2 top-level ops in the module, got "
                         << opCount;
      return signalPassFailure();
    }
    Operation *second = &module.getBody()->back();
    Operation *first = second->getPrevNode();

    llvm::outs() << first->getName().getStringRef() << " with attr "
                 << first->getDiscardableAttrDictionary();
    OperationEquivalence::Flags flags{};
    if (!first->hasAttr("strict_loc_check"))
      flags |= OperationEquivalence::IgnoreLocations;
    if (first->hasAttr("ignore_commutativity"))
      flags |= OperationEquivalence::IgnoreCommutativity;
    if (OperationEquivalence::isEquivalentTo(first, &module.getBody()->back(),
                                             flags))
      llvm::outs() << " compares equals.\n";
    else
      llvm::outs() << " compares NOT equals!\n";
  }
};

/// Test pass for `OperationEquivalence::isStructurallyEquivalentTo`. Compares
/// the first and last top-level ops of the module using a fresh
/// `StructuralCache`, then re-runs the comparison to exercise the cached path
/// and verifies both invocations agree.
struct TestStructuralEquivalencePass
    : public PassWrapper<TestStructuralEquivalencePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestStructuralEquivalencePass)

  StringRef getArgument() const final {
    return "test-operations-structural-equality";
  }
  StringRef getDescription() const final {
    return "Test OperationEquivalence::isStructurallyEquivalentTo.";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    int opCount = module.getBody()->getOperations().size();
    if (module->hasAttr("test.includes_setup")) {
      if (opCount < 2) {
        module.emitError()
            << "expected at least 2 top-level ops in the module, got "
            << opCount;
        return signalPassFailure();
      }
    } else if (opCount != 2) {
      module.emitError() << "expected 2 top-level ops in the module, got "
                         << opCount;
      return signalPassFailure();
    }
    Operation *second = &module.getBody()->back();
    Operation *first = second->getPrevNode();

    llvm::outs() << first->getName().getStringRef() << " with attr "
                 << first->getDiscardableAttrDictionary();
    OperationEquivalence::StructuralCache cache(&getContext());
    bool firstResult =
        OperationEquivalence::isStructurallyEquivalentTo(cache, *first,
                                                         *second);
    // Re-run to exercise the cached path; both queries must agree.
    bool cachedResult =
        OperationEquivalence::isStructurallyEquivalentTo(cache, *first,
                                                         *second);
    if (firstResult != cachedResult) {
      module.emitError() << "cache reuse changed result";
      return signalPassFailure();
    }
    if (firstResult)
      llvm::outs() << " structurally equals.\n";
    else
      llvm::outs() << " structurally NOT equals.\n";
  }
};
} // namespace

namespace mlir {
void registerTestOperationEqualPass() {
  PassRegistration<TestOperationEqualPass>();
}
void registerTestStructuralEquivalencePass() {
  PassRegistration<TestStructuralEquivalencePass>();
}
} // namespace mlir
