// RUN: mlir-opt %s -split-input-file --test-operations-structural-equality | FileCheck %s


// CHECK-LABEL: test.identical_no_region
// CHECK-SAME: structurally equals

"test.identical_no_region"() : () -> ()
"test.identical_no_region"() : () -> ()

// -----

// CHECK-LABEL: test.op_name_mismatch
// CHECK-SAME: structurally NOT equals

"test.op_name_mismatch"() : () -> ()
"test.op_name_mismatch2"() : () -> ()

// -----

// CHECK-LABEL: test.attr_value_mismatch
// CHECK-SAME: structurally NOT equals

"test.attr_value_mismatch"() { foo = "bar" } : () -> ()
"test.attr_value_mismatch"() { foo = "baz" } : () -> ()

// -----

// CHECK-LABEL: test.region_op_count_mismatch
// CHECK-SAME: structurally NOT equals

"test.region_op_count_mismatch"() ({
  "test.inner"() : () -> ()
  "test.inner"() : () -> ()
  "test.return"() : () -> ()
}) : () -> ()
"test.region_op_count_mismatch"() ({
  "test.inner"() : () -> ()
  "test.return"() : () -> ()
}) : () -> ()

// -----

// CHECK-LABEL: test.region_count_mismatch
// CHECK-SAME: structurally NOT equals

"test.region_count_mismatch"() ({
  "test.return"() : () -> ()
}) : () -> ()
"test.region_count_mismatch"() ({
  "test.return"() : () -> ()
}, {
  "test.return"() : () -> ()
}) : () -> ()

// -----

// CHECK-LABEL: test.identical_cfg
// CHECK-SAME: structurally equals

"test.identical_cfg"() ({
  ^bb0(%arg0 : i32, %arg1 : f32):
    "test.some_branching_op"(%arg1, %arg0) [^bb1, ^bb2] : (f32, i32) -> ()
  ^bb1(%arg2 : f32):
    "test.some_branching_op"() : () -> ()
  ^bb2(%arg3 : i32):
    "test.some_branching_op"() : () -> ()
  }) { attr = "foo" } : () -> ()
"test.identical_cfg"() ({
  ^bb0(%arg0 : i32, %arg1 : f32):
    "test.some_branching_op"(%arg1, %arg0) [^bb1, ^bb2] : (f32, i32) -> ()
  ^bb1(%arg2 : f32):
    "test.some_branching_op"() : () -> ()
  ^bb2(%arg3 : i32):
    "test.some_branching_op"() : () -> ()
  }) { attr = "foo" } : () -> ()

// -----

// CHECK-LABEL: test.successor_mismatch
// CHECK-SAME: structurally NOT equals

// The two CFGs differ only in which successor index points to which
// concrete-op block: lhs branches to (op_a, op_b), rhs branches to
// (op_b, op_a). The RPO mapping aligns lhs.bb1 -> rhs.bb1 (since both are
// visited at the same successor index from bb0), so the inner op-name
// difference is what makes this NOT equal.
"test.successor_mismatch"() ({
  ^bb0(%arg0 : i32):
    "test.some_branching_op"() [^bb1, ^bb2] : () -> ()
  ^bb1:
    "test.flavor_a"() : () -> ()
  ^bb2:
    "test.flavor_b"() : () -> ()
  }) : () -> ()
"test.successor_mismatch"() ({
  ^bb0(%arg0 : i32):
    "test.some_branching_op"() [^bb1, ^bb2] : () -> ()
  ^bb1:
    "test.flavor_b"() : () -> ()
  ^bb2:
    "test.flavor_a"() : () -> ()
  }) : () -> ()

// -----

// CHECK-LABEL: test.dataflow_match
// CHECK-SAME: structurally equals

"test.dataflow_match"() ({
  %0:2 = "test.producer"() : () -> (i32, i32)
  "test.consumer"(%0#0, %0#1) : (i32, i32) -> ()
  }) : () -> ()
"test.dataflow_match"() ({
  %0:2 = "test.producer"() : () -> (i32, i32)
  "test.consumer"(%0#0, %0#1) : (i32, i32) -> ()
  }) : () -> ()

// -----

// CHECK-LABEL: test.dataflow_operand_swap
// CHECK-SAME: structurally NOT equals

// Structural equivalence is operand-order strict; commutative reordering must
// not be considered equivalent.
"test.dataflow_operand_swap"() ({
  %0:2 = "test.producer"() : () -> (i32, i32)
  "test.consumer"(%0#0, %0#1) : (i32, i32) -> ()
  }) : () -> ()
"test.dataflow_operand_swap"() ({
  %0:2 = "test.producer"() : () -> (i32, i32)
  "test.consumer"(%0#1, %0#0) : (i32, i32) -> ()
  }) : () -> ()

// -----

// CHECK-LABEL: test.block_arg_type_mismatch
// CHECK-SAME: structurally NOT equals

"test.block_arg_type_mismatch"() ({
  ^bb0(%arg0 : i32):
    "test.return"() : () -> ()
  }) : () -> ()
"test.block_arg_type_mismatch"() ({
  ^bb0(%arg0 : f32):
    "test.return"() : () -> ()
  }) : () -> ()

// -----

// CHECK-LABEL: test.block_arg_count_mismatch
// CHECK-SAME: structurally NOT equals

"test.block_arg_count_mismatch"() ({
  ^bb0(%arg0 : i32, %arg1 : i32):
    "test.return"() : () -> ()
  }) : () -> ()
"test.block_arg_count_mismatch"() ({
  ^bb0(%arg0 : i32):
    "test.return"() : () -> ()
  }) : () -> ()

// -----

// CHECK-LABEL: test.result_type_mismatch
// CHECK-SAME: structurally NOT equals

%r1 = "test.result_type_mismatch"() : () -> (i32)
%r2 = "test.result_type_mismatch"() : () -> (f32)

// -----

// CHECK-LABEL: test.symbol_sym_name_ignored
// CHECK-SAME: structurally equals

// `sym_name` is a well-known symbol-reference attribute name and its value is
// skipped during structural comparison.
"test.symbol_sym_name_ignored"() { sym_name = "lhs" } : () -> ()
"test.symbol_sym_name_ignored"() { sym_name = "rhs" } : () -> ()

// -----

// CHECK-LABEL: test.symbol_function_ref_ignored
// CHECK-SAME: structurally equals

// `function_ref` is the other well-known symbol-reference attribute name.
"test.symbol_function_ref_ignored"() { function_ref = @target_a } : () -> ()
"test.symbol_function_ref_ignored"() { function_ref = @target_b } : () -> ()

// -----

// CHECK-LABEL: test.symbol_attr_name_diff_still_compared
// CHECK-SAME: structurally NOT equals

// Other attributes are still compared even if they reference symbols.
"test.symbol_attr_name_diff_still_compared"() { other = @target_a } : () -> ()
"test.symbol_attr_name_diff_still_compared"() { other = @target_b } : () -> ()

// -----

// CHECK-LABEL: test.nested_regions_match
// CHECK-SAME: structurally equals

"test.nested_regions_match"() ({
  "test.with_region"() ({
    "test.return"() : () -> ()
  }) : () -> ()
  "test.return"() : () -> ()
}) : () -> ()
"test.nested_regions_match"() ({
  "test.with_region"() ({
    "test.return"() : () -> ()
  }) : () -> ()
  "test.return"() : () -> ()
}) : () -> ()

// -----

// CHECK-LABEL: test.nested_regions_inner_op_diff
// CHECK-SAME: structurally NOT equals

"test.nested_regions_inner_op_diff"() ({
  "test.with_region"() ({
    "test.inner_a"() : () -> ()
    "test.return"() : () -> ()
  }) : () -> ()
  "test.return"() : () -> ()
}) : () -> ()
"test.nested_regions_inner_op_diff"() ({
  "test.with_region"() ({
    "test.inner_b"() : () -> ()
    "test.return"() : () -> ()
  }) : () -> ()
  "test.return"() : () -> ()
}) : () -> ()
