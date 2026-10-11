use ruda_kernel::{
    dsl as kernel_dsl,
    dsl::{InfoBuilder, prelude::*},
};
use rust_ascend::core::ir::{FloatKind, IntKind, UIntKind};
use rust_ascend::{
    compiler::{AscendCompiler, AscendOptions, AscendTarget, arguments},
    core::compiler::Compiler,
};

#[ruda(launch_unchecked, address_type = "dynamic")]
fn integer_loop(input: &Array<i64>, out: &mut Array<i64>, iterations: InputScalar) {
    if ABSOLUTE_POS >= out.len() {
        terminate!();
    }
    let mut value = input[ABSOLUTE_POS];
    for i in 0..iterations.get::<u32>() {
        if i % 2 == 0 {
            value += 2;
        } else {
            value -= 1;
        }
    }
    out[ABSOLUTE_POS] = value;
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn floating_modulo(input: &Array<f32>, out: &mut Array<f32>) {
    if ABSOLUTE_POS >= out.len() {
        terminate!();
    }
    out[ABSOLUTE_POS] = input[ABSOLUTE_POS] % 1.0;
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn dynamic_shape(input: &Tensor<i64>, out: &mut Array<i64>) {
    if ABSOLUTE_POS >= input.rank() {
        terminate!();
    }
    out[ABSOLUTE_POS] = i64::cast_from(input.shape(ABSOLUTE_POS));
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn shared_scratch(out: &mut Array<i32>) {
    let mut scratch = SharedMemory::<i32>::new(64usize);
    let unit = UNIT_POS as usize;
    if unit < scratch.len() {
        scratch[unit] = UNIT_POS as i32;
    }
    sync_ruda();
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = scratch[unit];
    }
}

#[test]
fn shared_memory_length_and_barrier_compile_with_packed_arguments() {
    let mut builder = KernelBuilder::default();
    AddressType::U64.register(&mut builder.scope);
    let out = builder.output_array(Type::new(IntKind::I32.into())).into();
    shared_scratch::expand(&mut builder.scope, out);
    let definition = builder.build(
        KernelSettings::default()
            .address_type(AddressType::U64)
            .ruda_dim(RudaDim::new_1d(64))
            .kernel_name("shared_scratch"),
    );
    let mut info = InfoBuilder::default();
    info.metadata.register_array(65, 65, AddressType::U64);
    let packed = info.finish(AddressType::U64);
    let definition = arguments::specialize(
        definition,
        &packed.data,
        packed.dynamic_metadata_offset,
        UIntKind::U64.into(),
    )
    .unwrap();
    let compiled = AscendCompiler
        .compile(
            definition,
            &AscendOptions {
                target: Some(AscendTarget::Ascend950DT),
                simt_grid: Some([2, 1, 1]),
                ..Default::default()
            },
            ExecutionMode::Unchecked,
            UIntKind::U64.into(),
        )
        .unwrap();
    assert!(compiled.source().contains("__ubuf__ int32_t"));
    assert!(compiled.source().contains("asc_syncthreads();"));
    assert_eq!(compiled.ub_bytes(), 256);
}

#[test]
fn dynamic_shape_queries_use_the_original_packed_metadata() {
    for address in [AddressType::U32, AddressType::U64] {
        let mut builder = KernelBuilder::default();
        address.register(&mut builder.scope);
        let input = builder.input_tensor(Type::new(IntKind::I64.into())).into();
        let out = builder.output_array(Type::new(IntKind::I64.into())).into();
        dynamic_shape::expand(&mut builder.scope, input, out);
        let definition = builder.build(
            KernelSettings::default()
                .address_type(address)
                .ruda_dim(RudaDim::new_1d(64))
                .kernel_name("dynamic_shape"),
        );
        let mut info = InfoBuilder::default();
        info.metadata
            .register_tensor(2, 77, 65, vec![5, 13].into(), vec![16, 1].into(), address);
        info.metadata.register_array(2, 2, address);
        let packed = info.finish(address);
        let definition = arguments::specialize(
            definition,
            &packed.data,
            packed.dynamic_metadata_offset,
            address.unsigned_type(),
        )
        .unwrap();
        let compiled = AscendCompiler
            .compile(
                definition,
                &AscendOptions {
                    target: Some(AscendTarget::Ascend950DT),
                    simt_grid: Some([1, 1, 1]),
                    ..Default::default()
                },
                ExecutionMode::Unchecked,
                address.unsigned_type(),
            )
            .unwrap();
        assert!(compiled.source().contains("const uint"));
        assert!(compiled.source().contains("[2] = {"));
        assert!(compiled.source().contains("0x0000000000000005ULL"));
        assert!(compiled.source().contains("0x000000000000000dULL"));
        assert_eq!(compiled.bindings()[0].bytes, 77 * 8);
    }
}

#[test]
fn actual_frontend_loop_preserves_integer_width_and_scalar_abi() {
    for address in [AddressType::U32, AddressType::U64] {
        let mut builder = KernelBuilder::default();
        address.register(&mut builder.scope);
        let input = builder.input_array(Type::new(IntKind::I64.into())).into();
        let out = builder.output_array(Type::new(IntKind::I64.into())).into();
        let iterations = <InputScalar as LaunchArg>::expand(
            &InputScalarCompilationArg::new(UIntKind::U32.into()),
            &mut builder,
        );
        integer_loop::expand(&mut builder.scope, input, out, iterations);
        let definition = builder.build(
            KernelSettings::default()
                .address_type(address)
                .ruda_dim(RudaDim::new_1d(64))
                .kernel_name("integer_loop"),
        );
        let mut info = InfoBuilder::default();
        info.scalars.push(7u32);
        info.metadata.register_array(65, 65, address);
        info.metadata.register_array(65, 65, address);
        let packed = info.finish(address);
        let definition = arguments::specialize(
            definition,
            &packed.data,
            packed.dynamic_metadata_offset,
            address.unsigned_type(),
        )
        .unwrap();
        let compiled = AscendCompiler
            .compile(
                definition,
                &AscendOptions {
                    target: Some(AscendTarget::Ascend950DT),
                    simt_grid: Some([2, 1, 1]),
                    ..Default::default()
                },
                ExecutionMode::Unchecked,
                address.unsigned_type(),
            )
            .unwrap();
        assert!(compiled.source().contains("for ("));
        assert!(compiled.source().contains("} else {"));
        assert!(compiled.source().contains("int64_t* b"));
        assert!(compiled.bindings().iter().all(|b| b.bytes == 520));
        assert_eq!(compiled.simt_dim(), Some([64, 1, 1]));
        assert_eq!(compiled.artifact_name(), "kernel.so");
    }
}

#[test]
fn modulo_frontend_uses_floating_remainder_not_integer_zero() {
    let mut builder = KernelBuilder::default();
    AddressType::U64.register(&mut builder.scope);
    let input = builder.input_array(Type::new(FloatKind::F32.into())).into();
    let out = builder
        .output_array(Type::new(FloatKind::F32.into()))
        .into();
    floating_modulo::expand(&mut builder.scope, input, out);
    let definition = builder.build(
        KernelSettings::default()
            .address_type(AddressType::U64)
            .ruda_dim(RudaDim::new_1d(64))
            .kernel_name("floating_modulo"),
    );
    let mut info = InfoBuilder::default();
    info.metadata.register_array(65, 65, AddressType::U64);
    info.metadata.register_array(65, 65, AddressType::U64);
    let packed = info.finish(AddressType::U64);
    let definition = arguments::specialize(
        definition,
        &packed.data,
        packed.dynamic_metadata_offset,
        UIntKind::U64.into(),
    )
    .unwrap();
    let compiled = AscendCompiler
        .compile(
            definition,
            &AscendOptions {
                target: Some(AscendTarget::Ascend950DT),
                simt_grid: Some([2, 1, 1]),
                ..Default::default()
            },
            ExecutionMode::Unchecked,
            UIntKind::U64.into(),
        )
        .unwrap();
    assert!(compiled.source().contains("fmodf("));
}

#[test]
fn typed_buffers_and_vector_items_keep_their_native_byte_widths() {
    use rust_ascend::{
        compiler::programs::{MapProgram, definition},
        core::ir::{ElemType, Operation, Operator},
    };
    let types = [
        Type::new(FloatKind::F32.into()),
        Type::new(FloatKind::F16.into()),
        Type::new(FloatKind::BF16.into()),
        Type::new(IntKind::I8.into()),
        Type::new(IntKind::I16.into()),
        Type::new(IntKind::I32.into()),
        Type::new(IntKind::I64.into()),
        Type::new(UIntKind::U8.into()),
        Type::new(UIntKind::U16.into()),
        Type::new(UIntKind::U32.into()),
        Type::new(UIntKind::U64.into()),
        Type::new(ElemType::Bool.into()),
        Type::Vector(FloatKind::F32.into(), 4),
    ];
    for ty in types {
        let mut kernel = definition(MapProgram::Copy);
        kernel.ruda_dim = RudaDim { x: 4, y: 4, z: 4 };
        for b in &mut kernel.buffers {
            b.ty = ty;
            b.size = Some(17);
        }
        for i in &mut kernel.body.instructions {
            match &mut i.operation {
                Operation::Operator(Operator::Index(o)) => {
                    o.list.ty = ty;
                    i.out.as_mut().unwrap().ty = ty;
                }
                Operation::Operator(Operator::IndexAssign(o)) => {
                    o.value.ty = ty;
                    i.out.as_mut().unwrap().ty = ty;
                }
                _ => panic!("unexpected copy program operation"),
            }
        }
        let compiled = AscendCompiler
            .compile(
                kernel,
                &AscendOptions {
                    target: Some(AscendTarget::Ascend950DT),
                    simt_grid: Some([2, 3, 4]),
                    ..Default::default()
                },
                ExecutionMode::Checked,
                UIntKind::U64.into(),
            )
            .unwrap();
        assert!(
            compiled
                .bindings()
                .iter()
                .all(|b| b.bytes == 17 * ty.size() as u64 && b.storage == ty.storage_type())
        );
        assert!(compiled.source().contains("<<<dim3(2,3,4), dim3(4,4,4)"));
    }
}

#[test]
#[ignore = "requires Linux, CANN 9.1 Bisheng and an exclusive Ascend 950DT"]
fn generic_kernels_compile_and_execute_on_950dt() -> Result<(), Box<dyn std::error::Error>> {
    use rust_ascend::{
        core::bytes::Bytes,
        runtime::{AscendRuntime, RuntimeOptions},
    };
    let toolkit = std::env::var_os("CANN_TOOLKIT_ROOT").ok_or("CANN_TOOLKIT_ROOT is required")?;
    let device = unsafe { AscendRuntime::initialize_exclusive(RuntimeOptions::new(toolkit))? };
    let client = AscendRuntime::client(&device);
    let input = [1i64 << 40, -(1i64 << 45), i64::MIN + 16, i64::MAX - 16];
    let data = client.create(Bytes::from_bytes_vec(
        input.iter().flat_map(|v| v.to_le_bytes()).collect(),
    ));
    let output = client.empty(input.len() * 8);
    unsafe {
        integer_loop::launch_unchecked::<AscendRuntime>(
            &client,
            RudaCount::Static(1, 1, 1),
            RudaDim::new_1d(64),
            AddressType::U64,
            ArrayArg::from_raw_parts(data, input.len()),
            ArrayArg::from_raw_parts(output.clone(), input.len()),
            InputScalar::new(7, UIntKind::U32),
        );
    }
    let bytes = client.read_one_unchecked(output);
    let actual: Vec<_> = bytes
        .chunks_exact(8)
        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(actual, input.map(|v| v + 5));
    let input = [1.5f32, -1.5, 2.0, -0.0];
    let data = client.create(Bytes::from_bytes_vec(
        input.iter().flat_map(|v| v.to_le_bytes()).collect(),
    ));
    let output = client.empty(input.len() * 4);
    unsafe {
        floating_modulo::launch_unchecked::<AscendRuntime>(
            &client,
            RudaCount::Static(1, 1, 1),
            RudaDim::new_1d(64),
            AddressType::U64,
            ArrayArg::from_raw_parts(data, input.len()),
            ArrayArg::from_raw_parts(output.clone(), input.len()),
        );
    }
    let bytes = client.read_one_unchecked(output);
    let actual: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()).to_bits())
        .collect();
    assert_eq!(actual, input.map(|v| (v % 1.0).to_bits()));
    Ok(())
}
