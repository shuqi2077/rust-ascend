use ruda_kernel::{
    dsl::{InfoBuilder, prelude::*},
    library::tensor::{AsViewExpand, AsViewMutExpand, layout::plain::PlainLayout},
};
use ruprim::elementwise::binary::numeric::{AddOp, kernel_binop, kernel_scalar_binop};
use rust_ascend::core::ir::FloatKind;
use rust_ascend::{
    compiler::{AscendCompiler, AscendOptions, AscendTarget, arguments},
    core::compiler::Compiler,
};

type E = Vector<f32, Const<1>>;

fn plain_input(
    builder: &mut KernelBuilder,
    info: &mut InfoBuilder,
    address: AddressType,
) -> <ruda_kernel::library::tensor::layout::linear::LinearView<E> as RudaType>::ExpandType {
    let array: NativeExpand<Array<E>> =
        builder.input_tensor(Type::new(FloatKind::F32.into())).into();
    let len = builder.scalar(address.unsigned_type()).into();
    if address == AddressType::U32 {
        info.scalars.push(65u32);
    } else {
        info.scalars.push(65u64);
    }
    info.metadata.register_tensor(2, 65, 65, vec![5, 13].into(), vec![13, 1].into(), address);
    let layout = PlainLayout::__expand_new(&mut builder.scope, len);
    array.__expand_view_method(&mut builder.scope, layout.into())
}

#[test]
fn actual_ruprim_binary_expansion_compiles_for_both_index_widths_and_inplace() {
    for address in [AddressType::U32, AddressType::U64] {
        for inplace in [false, true] {
            let mut builder = KernelBuilder::default();
            address.register(&mut builder.scope);
            let mut info = InfoBuilder::default();
            let lhs = plain_input(&mut builder, &mut info, address);
            let rhs = plain_input(&mut builder, &mut info, address);
            let output: NativeExpand<Array<E>> = if inplace {
                builder.inplace_output(0)
            } else {
                info.metadata.register_tensor(2, 65, 65, vec![5, 13].into(), vec![13, 1].into(), address);
                builder.output_tensor(Type::new(FloatKind::F32.into()))
            }
            .into();
            let len = builder.scalar(address.unsigned_type()).into();
            if address == AddressType::U32 {
                info.scalars.push(65u32);
            } else {
                info.scalars.push(65u64);
            }
            let layout = PlainLayout::__expand_new(&mut builder.scope, len);
            let output = output.__expand_view_mut_method(&mut builder.scope, layout.into());
            kernel_binop::expand::<f32, Const<1>, AddOp>(
                &mut builder.scope,
                1,
                lhs,
                rhs,
                output,
                FloatKind::F32.into(),
            );
            let definition = builder.build(
                KernelSettings::default()
                    .address_type(address)
                    .ruda_dim(RudaDim::new_1d(64))
                    .kernel_name("shared_add"),
            );
            let original = format!("{}", definition.body);
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
                        elements: 65,
                        ..Default::default()
                    },
                    ExecutionMode::Unchecked,
                    address.unsigned_type(),
                )
                .unwrap_or_else(|e| panic!("{e}\nOriginal RUDA IR:\n{original}"));
            assert!(compiled.source().contains("AscendC::Add("));
            assert_eq!(compiled.requires_initialized_outputs(), inplace);
            assert_eq!(compiled.bindings().len(), if inplace { 2 } else { 3 });
        }
    }
}

#[test]
fn ascend_uses_rudas_tensor_and_autodiff_contracts() {
    fn backend<B: ruda_tensor::Backend>() {}
    fn autodiff<B: ruda_tensor::backend::AutodiffBackend>() {}
    backend::<rust_ascend::Ascend>();
    autodiff::<rust_ascend::Autodiff<rust_ascend::Ascend>>();
}

#[test]
fn actual_scalar_kernel_preserves_fp32_bits_and_rejects_truncated_arguments() {
    for bits in [
        0x80000000u32,
        0x3f800000,
        0x7f800000,
        0x7fc12345,
        0x7f812345,
    ] {
        let mut builder = KernelBuilder::default();
        let address = AddressType::U32;
        address.register(&mut builder.scope);
        let mut info = InfoBuilder::default();
        let input = plain_input(&mut builder, &mut info, address);
        let scalar = <InputScalar as LaunchArg>::expand(
            &InputScalarCompilationArg::new(FloatKind::F32.into()),
            &mut builder,
        );
        info.scalars
            .push_raw(&bits.to_le_bytes(), FloatKind::F32.into());
        let output: NativeExpand<Array<E>> = builder.inplace_output(0).into();
        let len = builder.scalar(address.unsigned_type()).into();
        info.scalars.push(65u32);
        let layout = PlainLayout::__expand_new(&mut builder.scope, len);
        let output = output.__expand_view_mut_method(&mut builder.scope, layout.into());
        kernel_scalar_binop::expand::<f32, Const<1>, AddOp>(
            &mut builder.scope,
            1,
            input,
            scalar,
            output,
            FloatKind::F32.into(),
        );
        let definition = builder.build(
            KernelSettings::default()
                .address_type(address)
                .ruda_dim(RudaDim::new_1d(64))
                .kernel_name("shared_scalar"),
        );
        let original = format!("{}", definition.body);
        let packed = info.finish(address);
        assert!(
            arguments::specialize(definition.clone(), &[], 0, address.unsigned_type()).is_err()
        );
        assert!(
            arguments::specialize(
                definition.clone(),
                &packed.data,
                packed.dynamic_metadata_offset + 1,
                address.unsigned_type()
            )
            .is_err()
        );
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
                    elements: 65,
                    ..Default::default()
                },
                ExecutionMode::Unchecked,
                address.unsigned_type(),
            )
            .unwrap_or_else(|e| panic!("{e}\nOriginal RUDA IR:\n{original}"));
        assert!(compiled.source().contains(&format!("0x{bits:08x}U")));
    }
}
