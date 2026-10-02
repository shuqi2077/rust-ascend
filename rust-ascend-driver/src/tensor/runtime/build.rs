use super::{Result, error};
use rust_ascend_compiler::ascend::AscendKernel;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Duration,
};
use tempfile::TempDir;
use wait_timeout::ChildExt;

pub(super) struct Toolchain {
    compiler: PathBuf,
    linker: PathBuf,
    includes: Vec<PathBuf>,
    timeout: Duration,
    fingerprint: String,
}

impl Toolchain {
    pub fn new(root: &Path, timeout: Duration) -> Result<Self> {
        if timeout.is_zero() {
            return Err(error("compiler timeout must be positive"));
        }
        let compiler = root.join("bin/bisheng");
        let linker = root.join("bin/ld.lld");
        let mut digest = Sha256::new();
        for file in [&compiler, &linker] {
            digest.update(fs::read(file).map_err(error)?);
        }
        let base = [
            "aarch64-linux/asc/include",
            "x86_64-linux/asc/include",
            "include",
            "compiler/tikcpp/tikcfw",
        ]
        .iter()
        .map(|p| root.join(p))
        .find(|p| p.join("kernel_operator.h").is_file())
        .ok_or_else(|| error("installed CANN kernel_operator.h not found"))?;
        let includes = [
            base.clone(),
            base.join("interface"),
            base.join("impl"),
            base.join("adv_api"),
        ]
        .into_iter()
        .filter(|p| p.is_dir())
        .collect();
        Ok(Self {
            compiler,
            linker,
            includes,
            timeout,
            fingerprint: format!("{:x}", digest.finalize()),
        })
    }

    pub fn build(&self, kernel: &AscendKernel) -> Result<TempDir> {
        let work = self.build_source(kernel.source(), kernel.entrypoint(), false)?;
        let image = fs::read(work.path().join("kernel.o")).map_err(error)?;
        let manifest = format!(
            "{}object=kernel.o\nsource_sha256={:x}\nobject_sha256={:x}\ncompiler_sha256={}\n",
            kernel.build_contract(),
            Sha256::digest(kernel.source().as_bytes()),
            Sha256::digest(&image),
            self.fingerprint
        );
        fs::write(work.path().join("kernel.ruda"), manifest).map_err(error)?;
        Ok(work)
    }

    pub fn build_gemm(&self, key: &str, root: &Path) -> Result<()> {
        let spec = rust_ascend_kernels::Spec::all().into_iter().find(|s| s.key() == key)
            .ok_or_else(|| error("no Rust-authored matrix kernel for the GEMM contract"))?;
        let source = rust_ascend_kernels::emit(spec).map_err(error)?;
        let work = self.build_source(&source, &spec.entry(), true)?;
        let image = fs::read(work.path().join("kernel.o")).map_err(error)?;
        let manifest = format!(
            "schema=ruda.ascend.rust.bf16.v2\narch=dav-c310\nsoc=Ascend950DT\ncores=32\nkey={key}\nobject=kernel.o\nkernel_name={}\nsha256={:x}\nsource_sha256={:x}\ncompiler_sha256={}\nsource_language=rust-device-ir\nlowering=cann-c-intrinsics\n",
            spec.entry(), Sha256::digest(&image), Sha256::digest(source.as_bytes()), self.fingerprint
        );
        fs::write(work.path().join("kernel.ruda"), manifest).map_err(error)?;
        // Publish only a successfully compiled and linked artifact into this worker's cache.
        fs::rename(work.path(), root.join(key)).map_err(error)?;
        Ok(())
    }

    fn build_source(&self, source: &str, name: &str, matrix: bool) -> Result<TempDir> {
        let work = tempfile::Builder::new()
            .prefix("rust-ascend-jit-")
            .tempdir()
            .map_err(error)?;
        let root = work.path();
        fs::write(root.join("kernel.asc"), source).map_err(error)?;
        let mut compile = Command::new(&self.compiler);
        compile.args([
            "-x",
            "cce",
            "-std=c++20",
            "-O2",
            "--cce-aicore-only",
            "--cce-aicore-arch=dav-c310",
        ]);
        compile.arg(if matrix { "--cce-disable-vf-stack-reserved-ubuf" } else { "-ffp-contract=off" });
        for include in &self.includes {
            compile.arg("-I").arg(include);
        }
        compile
            .arg("-c")
            .arg(root.join("kernel.asc"))
            .arg("-o")
            .arg(root.join("kernel.rel.o"));
        self.run(compile, &root.join("compile.log"))?;
        let mut link = Command::new(&self.linker);
        link.args(["-m", "aicorelinux", "-Ttext", "0", "--no-mmap-output-file"])
            .arg(root.join("kernel.rel.o"))
            .arg("-o")
            .arg(root.join("kernel.o"));
        self.run(link, &root.join("link.log"))?;
        let image = fs::read(root.join("kernel.o")).map_err(error)?;
        if entrypoint(&image)? != name {
            return Err(error("linked CANN entrypoint differs from the IR kernel"));
        }
        Ok(work)
    }

    fn run(&self, mut command: Command, log: &Path) -> Result<()> {
        let file = fs::File::create(log).map_err(error)?;
        command
            .stdout(Stdio::from(file.try_clone().map_err(error)?))
            .stderr(Stdio::from(file));
        let mut child = command.spawn().map_err(error)?;
        let status = match child.wait_timeout(self.timeout).map_err(error)? {
            Some(status) => status,
            None => {
                child.kill().map_err(error)?;
                child.wait().map_err(error)?;
                return Err(error(format!("CANN compiler exceeded {:?}", self.timeout)));
            }
        };
        if !status.success() {
            return Err(error(format!(
                "CANN build failed ({status}): {}",
                fs::read_to_string(log).map_err(error)?
            )));
        }
        Ok(())
    }
}

fn entrypoint(image: &[u8]) -> Result<&str> {
    fn bytes(image: &[u8], at: usize, n: usize) -> Result<&[u8]> {
        image
            .get(
                at..at
                    .checked_add(n)
                    .ok_or_else(|| error("ELF offset overflow"))?,
            )
            .ok_or_else(|| error("truncated ELF"))
    }
    fn u16_at(image: &[u8], at: usize) -> Result<usize> {
        Ok(u16::from_le_bytes(bytes(image, at, 2)?.try_into().unwrap()) as usize)
    }
    fn u32_at(image: &[u8], at: usize) -> Result<usize> {
        Ok(u32::from_le_bytes(bytes(image, at, 4)?.try_into().unwrap()) as usize)
    }
    fn u64_at(image: &[u8], at: usize) -> Result<usize> {
        usize::try_from(u64::from_le_bytes(bytes(image, at, 8)?.try_into().unwrap())).map_err(error)
    }
    if image.get(..6) != Some(b"\x7fELF\x02\x01") || u16_at(image, 16)? != 2 {
        return Err(error("expected linked little-endian ELF64"));
    }
    let table = u64_at(image, 40)?;
    let count = u16_at(image, 60)?;
    let strings_index = u16_at(image, 62)?;
    if u16_at(image, 58)? != 64 || count == 0 || strings_index >= count {
        return Err(error("invalid ELF section table"));
    }
    bytes(image, table, count * 64)?;
    let strings_header = table + strings_index * 64;
    let strings = bytes(
        image,
        u64_at(image, strings_header + 24)?,
        u64_at(image, strings_header + 32)?,
    )?;
    let mut found = None;
    for i in 0..count {
        let start = u32_at(image, table + i * 64)?;
        let tail = strings
            .get(start..)
            .ok_or_else(|| error("invalid ELF section name"))?;
        let end = tail
            .iter()
            .position(|&b| b == 0)
            .ok_or_else(|| error("unterminated ELF section name"))?;
        let name = std::str::from_utf8(&tail[..end]).map_err(error)?;
        if let Some(name) = name.strip_prefix(".ascend.meta.") {
            if found.replace(name).is_some() || name.is_empty() {
                return Err(error("ambiguous CANN entrypoint"));
            }
        }
    }
    found.ok_or_else(|| error("missing CANN kernel metadata"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_missing_sdk_and_invalid_elf() {
        let dir = tempfile::tempdir().unwrap();
        assert!(Toolchain::new(dir.path(), Duration::from_secs(1)).is_err());
        for n in 0..128 {
            assert!(entrypoint(&vec![0; n]).is_err());
        }
    }
}
