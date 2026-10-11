use super::*;
use ruda_core::ir::*;

fn memory_views(mut kernel: KernelDefinition, values: bool) -> KernelDefinition {
    let last = kernel.body.instructions.iter().filter_map(|instruction| instruction.out)
        .filter_map(|value| match value.kind {
            VariableKind::LocalMut { id } | VariableKind::LocalConst { id } => Some(id),
            _ => None,
        }).max().unwrap_or(0);
    while kernel.body.allocator.new_local_index() <= last {}
    let instructions = std::mem::take(&mut kernel.body.instructions);
    for instruction in instructions {
        let (pointer, store) = match &instruction.operation {
            Operation::Operator(Operator::Index(op) | Operator::UncheckedIndex(op)) => (
                TypedPointer {
                    base: op.list, index: op.index,
                    ty: PointerType { pointee: op.list.ty, address_space: AddressSpace::Global,
                        access: PointerAccess::ReadOnly },
                    checked: matches!(&instruction.operation, Operation::Operator(Operator::Index(_))),
                    unroll_factor: op.unroll_factor,
                }, None),
            Operation::Operator(Operator::IndexAssign(op) | Operator::UncheckedIndexAssign(op)) => {
                let base = instruction.out.unwrap();
                (TypedPointer {
                    base, index: op.index,
                    ty: PointerType { pointee: base.ty, address_space: AddressSpace::Global,
                        access: PointerAccess::ReadWrite },
                    checked: matches!(&instruction.operation, Operation::Operator(Operator::IndexAssign(_))),
                    unroll_factor: op.unroll_factor,
                }, Some(op.value))
            }
            _ => { kernel.body.instructions.push(instruction); continue; }
        };
        let operation = if values {
            let address = *kernel.body.create_local(pointer.ty.value_type());
            kernel.body.register(Instruction::new(PointerOp::Address(pointer), address));
            match store {
                Some(value) => Operation::Pointer(PointerOp::Store { pointer: address, value, checked: pointer.checked }),
                None => Operation::Pointer(PointerOp::Load { pointer: address, checked: pointer.checked }),
            }
        } else {
            match store {
                Some(value) => Operation::TypedMemory(TypedMemory::Store { pointer, value }),
                None => Operation::TypedMemory(TypedMemory::Load(pointer)),
            }
        };
        kernel.body.instructions.push(Instruction {
            operation, out: if values && store.is_some() { None } else { instruction.out },
            modes: instruction.modes, source_loc: instruction.source_loc,
        });
    }
    kernel
}

fn compile(kernel: KernelDefinition, elements: u64, row_width: Option<u32>) -> AscendKernel {
    AscendCompiler.compile(kernel, &AscendOptions {
        target: Some(AscendTarget::Ascend950DT), elements, row_width,
        ..Default::default()
    }, ExecutionMode::Checked, UIntKind::U64.into()).unwrap()
}

#[test]
fn typed_and_first_class_pointer_maps_preserve_common_ir_results() {
    use super::programs::{definition, MapProgram};
    let input = [0., -0., 1.5, -3., f32::INFINITY, f32::NEG_INFINITY, f32::from_bits(0x7fc01234)];
    for values in [false, true] {
        let kernel = memory_views(definition(MapProgram::Copy), values);
        compile(kernel.clone(), input.len() as u64, None);
        let output = super::tests::evaluate(kernel, input.len(), [&input, &[], &[]]);
        assert_eq!(output[0].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            input.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    }
}

#[test]
fn typed_and_first_class_pointer_rows_preserve_vector_programs() {
    use super::row_programs::{definition, RowProgram};
    for program in [RowProgram::Softmax, RowProgram::RmsNorm, RowProgram::LayerNorm] {
        let original = definition(program, 32, 1e-5).unwrap();
        let expected = compile(original.clone(), 96, Some(32));
        for values in [false, true] {
            let actual = compile(memory_views(original.clone(), values), 96, Some(32));
            assert_eq!(actual.source(), expected.source());
            assert_eq!(actual.bindings(), expected.bindings());
            assert_eq!(actual.ub_bytes(), expected.ub_bytes());
        }
    }
}

#[test]
fn typed_memory_rejects_write_access_to_readonly_bindings() {
    let mut kernel = memory_views(super::programs::definition(super::programs::MapProgram::Copy), false);
    for instruction in &mut kernel.body.instructions {
        if let Operation::TypedMemory(TypedMemory::Load(pointer)) = &mut instruction.operation {
            pointer.ty.access = PointerAccess::ReadWrite;
        }
    }
    let result = AscendCompiler.compile(kernel, &AscendOptions {
        target: Some(AscendTarget::Ascend950DT), elements: 7, ..Default::default()
    }, ExecutionMode::Checked, UIntKind::U64.into());
    assert!(matches!(result, Err(CompilationError::Validation { .. })));
}
