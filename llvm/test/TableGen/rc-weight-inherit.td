// RUN: llvm-tblgen -gen-register-info -register-info-debug -I %p/../../include %s -o - 2>&1 >/dev/null | FileCheck %s

// A register class synthesized by TableGen inherits the Weight of its nearest
// super-class. Without that, its weight is the sum of the register unit
// weights of its members.

include "llvm/Target/Target.td"

def TestTarget : Target;

def lo16 : SubRegIndex<16, 0>;
def hi16 : SubRegIndex<16, 16>;
def sub0 : SubRegIndex<32, 0>;
def sub1 : SubRegIndex<32, 32>;

// Each 32-bit register has two 16-bit halves, so it has two register units.
foreach I = 0...3 in {
  def L#I : Register<"l"#I>;
  def H#I : Register<"h"#I>;
  def R#I : Register<"r"#I> {
    let SubRegs = [!cast<Register>("L"#I), !cast<Register>("H"#I)];
    let SubRegIndices = [lo16, hi16];
    let CoveredBySubRegs = 1;
  }
}

foreach I = 0...2 in {
  def D#I : RegisterWithSubRegs<"d"#I, [!cast<Register>("R"#I),
                                        !cast<Register>("R"#!add(I, 1))]> {
    let SubRegIndices = [sub0, sub1];
    let CoveredBySubRegs = 1;
  }
}

def GPR16 : RegisterClass<"", [i16], 16, (add (sequence "L%u", 0, 3),
                                             (sequence "H%u", 0, 3))>;

// The pressure of one 32-bit register is 1, not 2.
def GPR32 : RegisterClass<"", [i32], 32, (sequence "R%u", 0, 3)> {
  let Weight = 1;
}

def GPR32_Lo : RegisterClass<"", [i32], 32, (add R0, R1)> {
  let Weight = 1;
}

def GPR64 : RegisterClass<"", [i64], 64, (sequence "D%u", 0, 2)> {
  let Weight = 2;
}

// CHECK-LABEL: RegisterClass GPR32:
// CHECK:       Weight: 1
// CHECK-LABEL: RegisterClass GPR64:
// CHECK:       Weight: 2
// CHECK-LABEL: RegisterClass GPR64_with_sub0_in_GPR32_Lo:
// CHECK:       Weight: 2
// CHECK-NEXT:  BaseClassOrder:
// CHECK-NEXT:  Regs: D0 D1{{$}}
// CHECK-LABEL: RegisterClass GPR64_with_sub1_in_GPR32_Lo:
// CHECK:       Weight: 2
// CHECK-NEXT:  BaseClassOrder:
// CHECK-NEXT:  Regs: D0{{$}}
