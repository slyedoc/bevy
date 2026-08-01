//! Minimal libslang FFI: runtime Slang→SPIR-V for the RT stages that compose
//! runtime-swappable modules.
//!
//! Most RT stages ship as Slang-precompiled SPIR-V blobs (regen commands in
//! each `.slang` header). Two things can't be precompiled: the primary miss
//! (it composes the app-swappable `custom_sky` module) and downstream
//! [`SolariChitSource::Slang`](crate::gpu::rt_pipeline::SolariChitSource)
//! closest-hits. Those compile here, through the deprecated-but-exported
//! `sp*` compile-request C API — plain C symbols, no COM vtables — dlopen'd
//! from the pinned Slang toolchain.
//!
//! The library is resolved from `$SLANG_DIR/lib/libslang.so` (default
//! `SLANG_DIR=/mnt/code/f/slang`, the same install whose `bin/slangc`
//! produced the checked-in blobs — keep them the same version: modules
//! compiled by mismatched compilers may disagree on layout).
//!
//! Module composition uses a per-compile temp directory on the request's
//! search path — identical resolution semantics to the `slangc` CLI the
//! precompiled blobs are built with.

#![allow(unsafe_code)]

use std::ffi::{c_char, c_int, c_void, CString};
use std::sync::Mutex;

/// RT stage of a runtime-compiled entry point.
#[derive(Clone, Copy, Debug)]
pub enum SlangRtStage {
    RayGeneration,
    AnyHit,
    ClosestHit,
    Miss,
}

impl SlangRtStage {
    /// `SlangStage` values from slang.h.
    fn raw(self) -> u32 {
        match self {
            SlangRtStage::RayGeneration => 7,
            SlangRtStage::AnyHit => 9,
            SlangRtStage::ClosestHit => 10,
            SlangRtStage::Miss => 11,
        }
    }
}

/// `SlangCompileTarget` from slang.h.
const SLANG_SPIRV: c_int = 6;
/// `SlangSourceLanguage` from slang.h.
const SLANG_SOURCE_LANGUAGE_SLANG: c_int = 1;

/// The `sp*` entry points this module owns, resolved once from libslang.so.
/// `SlangSession` / `SlangCompileRequest` are opaque.
struct Api {
    // Keeps the dlopen'd library resident for the fn pointers' lifetime.
    _lib: libloading::Library,
    session: *mut c_void,
    create_compile_request: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    destroy_compile_request: unsafe extern "C" fn(*mut c_void),
    add_code_gen_target: unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
    add_search_path: unsafe extern "C" fn(*mut c_void, *const c_char),
    add_translation_unit: unsafe extern "C" fn(*mut c_void, c_int, *const c_char) -> c_int,
    add_translation_unit_source_string:
        unsafe extern "C" fn(*mut c_void, c_int, *const c_char, *const c_char),
    add_entry_point: unsafe extern "C" fn(*mut c_void, c_int, *const c_char, u32) -> c_int,
    compile: unsafe extern "C" fn(*mut c_void) -> i32,
    get_diagnostic_output: unsafe extern "C" fn(*mut c_void) -> *const c_char,
    get_entry_point_code: unsafe extern "C" fn(*mut c_void, c_int, *mut usize) -> *const c_void,
}

// SAFETY: the session pointer is only ever dereferenced by libslang calls made
// under the `SLANG` mutex below, so cross-thread moves are serialized.
unsafe impl Send for Api {}

/// Global compiler state. Slang sessions are not thread-safe; every use is
/// serialized under this lock (registration-time validation runs on the main
/// thread, pipeline builds on the render thread). A load failure is cached —
/// it's a broken/missing toolchain install, not transient.
static SLANG: Mutex<Option<Result<Api, String>>> = Mutex::new(None);

fn load_api() -> Result<Api, String> {
    let dir = std::env::var("SLANG_DIR").unwrap_or_else(|_| "/mnt/code/f/slang".to_string());
    let path = format!("{dir}/lib/libslang.so");
    // SAFETY: libslang runs arbitrary initialization like any dlopen'd library;
    // it's the pinned toolchain we also run `slangc` from.
    let lib = unsafe { libloading::Library::new(&path) }
        .map_err(|e| format!("dlopen {path}: {e} (set SLANG_DIR to the Slang install root)"))?;

    macro_rules! sym {
        ($name:literal) => {
            // SAFETY: signature transcribed from the pinned install's
            // include/slang-deprecated.h.
            *unsafe { lib.get($name) }.map_err(|e| format!("{path}: {}: {e}", $name.escape_ascii()))?
        };
    }
    let create_session: unsafe extern "C" fn(*const c_char) -> *mut c_void = sym!(b"spCreateSession");
    let api = Api {
        create_compile_request: sym!(b"spCreateCompileRequest"),
        destroy_compile_request: sym!(b"spDestroyCompileRequest"),
        add_code_gen_target: sym!(b"spAddCodeGenTarget"),
        add_search_path: sym!(b"spAddSearchPath"),
        add_translation_unit: sym!(b"spAddTranslationUnit"),
        add_translation_unit_source_string: sym!(b"spAddTranslationUnitSourceString"),
        add_entry_point: sym!(b"spAddEntryPoint"),
        compile: sym!(b"spCompile"),
        get_diagnostic_output: sym!(b"spGetDiagnosticOutput"),
        get_entry_point_code: sym!(b"spGetEntryPointCode"),
        // SAFETY: the argument is a deprecated no-op config string; null is the
        // documented default.
        session: unsafe { create_session(core::ptr::null()) },
        _lib: lib,
    };
    if api.session.is_null() {
        return Err(format!("{path}: spCreateSession returned null"));
    }
    Ok(api)
}

/// Compile one RT-stage entry point to SPIR-V words.
///
/// `entry_source` is a translation unit named `entry_file`; `entry_name` is the
/// entry function inside it (the emitted `OpEntryPoint` is renamed `"main"`, as
/// with every slangc-compiled stage). `modules` are `(module_name, source)`
/// pairs made importable to the entry (and to each other) — the same closure a
/// `slangc` invocation would resolve from files next to the entry.
pub fn compile_rt_slang(
    entry_file: &str,
    entry_source: &str,
    entry_name: &str,
    stage: SlangRtStage,
    modules: &[(&str, &str)],
) -> Result<Vec<u32>, String> {
    let mut guard = SLANG.lock().unwrap();
    let api = match guard.get_or_insert_with(load_api) {
        Ok(api) => api,
        Err(e) => return Err(e.clone()),
    };

    // Imports resolve through a per-compile module directory on the search
    // path — exactly how the CLI resolves `import x;` as `x.slang` beside the
    // entry file.
    let module_dir = std::env::temp_dir().join(format!(
        "bevy-solari-slang-{}-{:p}",
        std::process::id(),
        entry_source.as_ptr()
    ));
    std::fs::create_dir_all(&module_dir).map_err(|e| format!("{}: {e}", module_dir.display()))?;
    for (name, source) in modules {
        let path = module_dir.join(format!("{name}.slang"));
        std::fs::write(&path, source).map_err(|e| format!("{}: {e}", path.display()))?;
    }

    let result = compile_with_request(api, &module_dir, entry_file, entry_source, entry_name, stage);
    let _ = std::fs::remove_dir_all(&module_dir);
    result
}

fn compile_with_request(
    api: &Api,
    module_dir: &std::path::Path,
    entry_file: &str,
    entry_source: &str,
    entry_name: &str,
    stage: SlangRtStage,
) -> Result<Vec<u32>, String> {
    let cstr = |s: &str, what: &str| {
        CString::new(s).map_err(|_| format!("{entry_file}: interior NUL in {what}"))
    };
    let search = cstr(&module_dir.to_string_lossy(), "module dir path")?;
    let tu_name = cstr(entry_file.trim_end_matches(".slang"), "entry file name")?;
    let tu_path = cstr(entry_file, "entry file name")?;
    let source = cstr(entry_source, "shader source")?;
    let entry = cstr(entry_name, "entry point name")?;

    // SAFETY: request/session pointers come from the same live libslang; every
    // passed pointer outlives the call (CStrings live to the end of scope), and
    // the request is destroyed on every path below.
    unsafe {
        let req = (api.create_compile_request)(api.session);
        if req.is_null() {
            return Err(format!("{entry_file}: spCreateCompileRequest returned null"));
        }
        // Everything below mirrors `slangc <entry> -target spirv -entry <name>`
        // with defaults — the checked-in blobs' exact recipe.
        (api.add_code_gen_target)(req, SLANG_SPIRV);
        (api.add_search_path)(req, search.as_ptr());
        let tu = (api.add_translation_unit)(req, SLANG_SOURCE_LANGUAGE_SLANG, tu_name.as_ptr());
        (api.add_translation_unit_source_string)(req, tu, tu_path.as_ptr(), source.as_ptr());
        (api.add_entry_point)(req, tu, entry.as_ptr(), stage.raw());

        let compile_result = (api.compile)(req);
        let diagnostics = {
            let ptr = (api.get_diagnostic_output)(req);
            if ptr.is_null() {
                String::new()
            } else {
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        if compile_result < 0 {
            (api.destroy_compile_request)(req);
            return Err(format!("{entry_file}: slang compile failed:\n{diagnostics}"));
        }
        if !diagnostics.trim().is_empty() {
            tracing::warn!("{entry_file}: slang diagnostics:\n{diagnostics}");
        }

        let mut size: usize = 0;
        let code = (api.get_entry_point_code)(req, 0, &mut size);
        if code.is_null() || size < 20 || size % 4 != 0 {
            (api.destroy_compile_request)(req);
            return Err(format!(
                "{entry_file}: slang produced no/truncated code ({size} bytes)"
            ));
        }
        let bytes = std::slice::from_raw_parts(code.cast::<u8>(), size);
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (api.destroy_compile_request)(req);

        if words[0] != 0x0723_0203 {
            return Err(format!("{entry_file}: slang output is not SPIR-V"));
        }
        Ok(words)
    }
}

