//! Minimal libslang FFI: runtime Slang→SPIR-V for every solari GPU stage.
//!
//! All stages — the RT pipeline libraries, downstream
//! [`SolariHitGroupDef`](crate::gpu::rt_pipeline::SolariHitGroupDef) hits, and
//! the NRC compute kernels — compile from `include_str!` source at pipeline
//! build, through the deprecated-but-exported `sp*` compile-request C API —
//! plain C symbols, no COM vtables — dlopen'd from the pinned Slang toolchain.
//! One compiler compiles every module in a link, so cross-module layout
//! agreement (payload structs, the MLP constants) holds by construction, and
//! variant axes (`defines`, swappable modules like `custom_sky`) are ordinary
//! build inputs instead of a checked-in blob per combination.
//!
//! The library is resolved from `$SLANG_DIR/lib/libslang.so` (default
//! `SLANG_DIR=/mnt/code/f/slang`).
//!
//! Module composition uses a per-compile temp directory on the request's
//! search path — identical resolution semantics to `import x;` finding
//! `x.slang` beside the entry file under the `slangc` CLI.

#![allow(unsafe_code)]

use std::ffi::{c_char, c_int, c_uint, c_void, CString};
use std::sync::Mutex;

/// A compiled entry point: the SPIR-V plus the program's reflected
/// descriptor-bound global parameters. The parameter table lets dispatch-side
/// binding tables be assembled BY NAME against the shader's own layout
/// instead of hand-ordered ("must match the `[[vk::binding]]` table")
/// contracts — see `NrcKernel::push_slots`.
pub struct CompiledShader {
    pub spirv: Vec<u32>,
    /// `(parameter name, descriptor set, binding)` per global parameter, in
    /// reflection order. Reflection covers the declared layout — a parameter
    /// DCE'd out of the SPIR-V still appears here.
    pub bindings: Vec<(String, u32, u32)>,
}

/// Stage of a runtime-compiled entry point.
#[derive(Clone, Copy, Debug)]
pub enum SlangRtStage {
    RayGeneration,
    AnyHit,
    ClosestHit,
    Miss,
    Compute,
}

impl SlangRtStage {
    /// `SlangStage` values from slang.h.
    fn raw(self) -> u32 {
        match self {
            SlangRtStage::RayGeneration => 7,
            SlangRtStage::AnyHit => 9,
            SlangRtStage::ClosestHit => 10,
            SlangRtStage::Miss => 11,
            SlangRtStage::Compute => 6,
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
    add_preprocessor_define: unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char),
    find_capability: unsafe extern "C" fn(*mut c_void, *const c_char) -> i32,
    add_target_capability: unsafe extern "C" fn(*mut c_void, c_int, i32),
    get_reflection: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    reflection_parameter_count: unsafe extern "C" fn(*mut c_void) -> c_uint,
    reflection_parameter_by_index: unsafe extern "C" fn(*mut c_void, c_uint) -> *mut c_void,
    parameter_binding_index: unsafe extern "C" fn(*mut c_void) -> c_uint,
    parameter_binding_space: unsafe extern "C" fn(*mut c_void) -> c_uint,
    variable_layout_variable: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    variable_name: unsafe extern "C" fn(*mut c_void) -> *const c_char,
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
        add_preprocessor_define: sym!(b"spAddPreprocessorDefine"),
        find_capability: sym!(b"spFindCapability"),
        add_target_capability: sym!(b"spAddTargetCapability"),
        get_reflection: sym!(b"spGetReflection"),
        reflection_parameter_count: sym!(b"spReflection_GetParameterCount"),
        reflection_parameter_by_index: sym!(b"spReflection_GetParameterByIndex"),
        parameter_binding_index: sym!(b"spReflectionParameter_GetBindingIndex"),
        parameter_binding_space: sym!(b"spReflectionParameter_GetBindingSpace"),
        variable_layout_variable: sym!(b"spReflectionVariableLayout_GetVariable"),
        variable_name: sym!(b"spReflectionVariable_GetName"),
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
/// entry function inside it (the emitted `OpEntryPoint` is renamed `"main"`).
/// `modules` are `(module_name, source)` pairs made importable to the entry
/// (and to each other) — the same closure a `slangc` invocation would resolve
/// from files next to the entry. `defines` are `(key, value)` preprocessor
/// defines applied to the entry translation unit (`slangc -D key=value`) — the
/// compile-out feature axes like raygen's `SOLARI_SHADER_CLOCK`. `capabilities`
/// are target capability atoms (`slangc -capability x`): declaring the
/// target's capability SET restricts which SPIR-V flavor gets emitted — e.g.
/// `spvShaderInvocationReorderNV` pins SER to the NV capability/extension the
/// device enables; without it slang free-chooses and emits the EXT flavor.
pub fn compile_rt_slang(
    entry_file: &str,
    entry_source: &str,
    entry_name: &str,
    stage: SlangRtStage,
    modules: &[(&str, &str)],
    defines: &[(&str, &str)],
    capabilities: &[&str],
) -> Result<CompiledShader, String> {
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

    let result = compile_with_request(
        api,
        &module_dir,
        entry_file,
        entry_source,
        entry_name,
        stage,
        defines,
        capabilities,
    );
    let _ = std::fs::remove_dir_all(&module_dir);
    result
}

#[allow(clippy::too_many_arguments)]
fn compile_with_request(
    api: &Api,
    module_dir: &std::path::Path,
    entry_file: &str,
    entry_source: &str,
    entry_name: &str,
    stage: SlangRtStage,
    defines: &[(&str, &str)],
    capabilities: &[&str],
) -> Result<CompiledShader, String> {
    let cstr = |s: &str, what: &str| {
        CString::new(s).map_err(|_| format!("{entry_file}: interior NUL in {what}"))
    };
    let search = cstr(&module_dir.to_string_lossy(), "module dir path")?;
    let tu_name = cstr(entry_file.trim_end_matches(".slang"), "entry file name")?;
    let tu_path = cstr(entry_file, "entry file name")?;
    let source = cstr(entry_source, "shader source")?;
    let entry = cstr(entry_name, "entry point name")?;
    let defines: Vec<(CString, CString)> = defines
        .iter()
        .map(|(k, v)| Ok((cstr(k, "define key")?, cstr(v, "define value")?)))
        .collect::<Result<_, String>>()?;
    let capabilities: Vec<CString> = capabilities
        .iter()
        .map(|c| cstr(c, "capability name"))
        .collect::<Result<_, String>>()?;

    // SAFETY: request/session pointers come from the same live libslang; every
    // passed pointer outlives the call (CStrings live to the end of scope), and
    // the request is destroyed on every path below.
    unsafe {
        let req = (api.create_compile_request)(api.session);
        if req.is_null() {
            return Err(format!("{entry_file}: spCreateCompileRequest returned null"));
        }
        // Everything below mirrors
        // `slangc <entry> -target spirv -entry <name> [-D k=v] [-capability c]`
        // with defaults.
        let target = (api.add_code_gen_target)(req, SLANG_SPIRV);
        (api.add_search_path)(req, search.as_ptr());
        for (k, v) in &defines {
            (api.add_preprocessor_define)(req, k.as_ptr(), v.as_ptr());
        }
        for cap in &capabilities {
            // SLANG_CAPABILITY_UNKNOWN = 0.
            let id = (api.find_capability)(api.session, cap.as_ptr());
            if id <= 0 {
                (api.destroy_compile_request)(req);
                return Err(format!(
                    "{entry_file}: unknown slang capability {:?}",
                    cap.to_string_lossy()
                ));
            }
            (api.add_target_capability)(req, target, id);
        }
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

        // Global-parameter reflection, copied out before the request (which
        // owns the reflection blob) is destroyed.
        let mut bindings = Vec::new();
        let reflection = (api.get_reflection)(req);
        if !reflection.is_null() {
            for i in 0..(api.reflection_parameter_count)(reflection) {
                let param = (api.reflection_parameter_by_index)(reflection, i);
                if param.is_null() {
                    continue;
                }
                let var = (api.variable_layout_variable)(param);
                let name_ptr = if var.is_null() {
                    core::ptr::null()
                } else {
                    (api.variable_name)(var)
                };
                if name_ptr.is_null() {
                    continue;
                }
                let name = std::ffi::CStr::from_ptr(name_ptr)
                    .to_string_lossy()
                    .into_owned();
                bindings.push((
                    name,
                    (api.parameter_binding_space)(param),
                    (api.parameter_binding_index)(param),
                ));
            }
        }
        (api.destroy_compile_request)(req);

        if words[0] != 0x0723_0203 {
            return Err(format!("{entry_file}: slang output is not SPIR-V"));
        }
        Ok(CompiledShader {
            spirv: words,
            bindings,
        })
    }
}

