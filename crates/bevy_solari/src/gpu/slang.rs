//! Minimal libslang FFI: runtime Slang→SPIR-V for every solari GPU stage.
//!
//! All stages — the RT pipeline libraries, downstream
//! [`SolariHitGroupDef`](crate::gpu::rt_pipeline::SolariHitGroupDef) hits, and
//! the NRC compute kernels — compile from `include_str!` source at pipeline
//! build, through the modern COM compile API (`IGlobalSession` → `ISession` →
//! `IModule`/`IComponentType`), dlopen'd from the pinned Slang toolchain.
//! One compiler compiles every module in a link, so cross-module layout
//! agreement (payload structs, the MLP constants) holds by construction, and
//! variant axes (`defines`, swappable modules like `custom_sky`) are ordinary
//! build inputs instead of a checked-in blob per combination. Modules load
//! straight from source strings (`loadModuleFromSourceString`) — imports
//! resolve against the session's loaded modules, no files touched. The entry
//! point's stage comes from its `[shader("...")]` attribute.
//!
//! The COM surface is transcribed from the pinned install's `slang.h`:
//! interfaces are vtable pointers ([`com_slot`]), descriptor structs are
//! `#[repr(C)]` mirrors, and the reflection walk uses the plain-C
//! `spReflection_*` symbols (not deprecated) against
//! `IComponentType::getLayout`'s `ProgramLayout`.
//!
//! Results are cached on disk keyed by the full compile input (compiler
//! build tag + sources + modules + defines + capabilities), so warm startups
//! skip compilation entirely; the stored key is verified byte-for-byte on
//! load, making filename-hash collisions harmless.
//!
//! The library is resolved from `$SLANG_DIR/lib/libslang.so` (default
//! `SLANG_DIR=/mnt/code/f/slang`).

#![allow(unsafe_code)]

use std::ffi::{c_char, c_uint, c_void, CString};
use std::path::PathBuf;
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

/// `SlangCompileTarget` from slang.h.
const SLANG_SPIRV: i32 = 6;
/// `CompilerOptionName::Capability` — `intValue0` is a `SlangCapabilityID`.
const COMPILER_OPTION_CAPABILITY: i32 = 39;
/// `CompilerOptionName::DebugInformation` — `intValue0` is a
/// `SlangDebugInfoLevel`.
const COMPILER_OPTION_DEBUG_INFORMATION: i32 = 44;
/// `CompilerOptionName::DebugInfoIncludeSource` — embed the source text into
/// the debug info regardless of level.
const COMPILER_OPTION_DEBUG_INFO_INCLUDE_SOURCE: i32 = 157;
/// `CompilerOptionName::VulkanUseEntryPointName` — emit `OpEntryPoint` with
/// the source entry name instead of `"main"`. Pipeline stages pass the real
/// name as `pName`; distinct names also keep linked RT pipeline libraries'
/// debug records from all colliding on `"main"`.
const COMPILER_OPTION_VULKAN_USE_ENTRY_POINT_NAME: i32 = 52;
/// `SLANG_DEBUG_INFO_LEVEL_MAXIMAL`.
const DEBUG_INFO_LEVEL_MAXIMAL: i32 = 3;
/// `CompilerOptionValueKind::Int`.
const OPTION_KIND_INT: i32 = 0;
/// `kDefaultTargetFlags` (= GENERATE_SPIRV_DIRECTLY).
const DEFAULT_TARGET_FLAGS: u32 = 1 << 10;
/// `SLANG_MATRIX_LAYOUT_ROW_MAJOR` (the `SessionDesc` C++ default).
const MATRIX_LAYOUT_ROW_MAJOR: u32 = 1;

// COM vtable slot indices, counted off the pinned slang.h's virtual method
// order (ISlangUnknown occupies 0..=2 everywhere; IModule and IEntryPoint
// extend IComponentType, whose 14 methods occupy 3..=16).
const IGLOBAL_SESSION_CREATE_SESSION: usize = 3;
const ISESSION_CREATE_COMPOSITE_COMPONENT_TYPE: usize = 6;
const ISESSION_LOAD_MODULE_FROM_SOURCE_STRING: usize = 20;
const ICOMPONENT_TYPE_GET_LAYOUT: usize = 4;
const ICOMPONENT_TYPE_GET_ENTRY_POINT_CODE: usize = 6;
const ICOMPONENT_TYPE_LINK: usize = 10;
const IMODULE_FIND_ENTRY_POINT_BY_NAME: usize = 17;
const IBLOB_GET_BUFFER_POINTER: usize = 3;
const IBLOB_GET_BUFFER_SIZE: usize = 4;
const IUNKNOWN_RELEASE: usize = 2;

/// `slang::TargetDesc` mirror (slang.h; field order and types must match).
#[repr(C)]
struct TargetDesc {
    structure_size: usize,
    format: i32,
    profile: u32,
    flags: u32,
    floating_point_mode: u32,
    line_directive_mode: u32,
    force_glsl_scalar_buffer_layout: bool,
    compiler_option_entries: *const CompilerOptionEntry,
    compiler_option_entry_count: u32,
}

/// `slang::SessionDesc` mirror.
#[repr(C)]
struct SessionDesc {
    structure_size: usize,
    targets: *const TargetDesc,
    target_count: i64,
    flags: u32,
    default_matrix_layout_mode: u32,
    search_paths: *const *const c_char,
    search_path_count: i64,
    preprocessor_macros: *const PreprocessorMacroDesc,
    preprocessor_macro_count: i64,
    file_system: *mut c_void,
    enable_effect_annotations: bool,
    allow_glsl_syntax: bool,
    compiler_option_entries: *const CompilerOptionEntry,
    compiler_option_entry_count: u32,
    skip_spirv_validation: bool,
}

#[repr(C)]
struct PreprocessorMacroDesc {
    name: *const c_char,
    value: *const c_char,
}

#[repr(C)]
struct CompilerOptionEntry {
    name: i32,
    value: CompilerOptionValue,
}

#[repr(C)]
struct CompilerOptionValue {
    kind: i32,
    int_value0: i32,
    int_value1: i32,
    string_value0: *const c_char,
    string_value1: *const c_char,
}

/// Fetch a COM interface's vtable slot.
///
/// # Safety
/// `obj` must be a live slang COM interface pointer and `index` a valid slot
/// for its concrete interface.
unsafe fn com_slot(obj: *mut c_void, index: usize) -> *const c_void {
    unsafe { *(*(obj as *mut *const *const c_void)).add(index) }
}

/// Owned COM reference: releases on drop (null is a no-op).
struct ComPtr(*mut c_void);

impl Drop for ComPtr {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: the pointer is a live interface this ComPtr owns a
            // reference to; release is ISlangUnknown slot 2 on every one.
            unsafe {
                let release: unsafe extern "C" fn(*mut c_void) -> u32 =
                    core::mem::transmute(com_slot(self.0, IUNKNOWN_RELEASE));
                release(self.0);
            }
        }
    }
}

/// The libslang surface this module owns: the COM global session plus the
/// plain-C symbols (capability lookup + reflection — not deprecated).
struct Api {
    // Keeps the dlopen'd library resident for the fn pointers' lifetime.
    _lib: libloading::Library,
    /// `slang::IGlobalSession*`. The C `sp*` reflection/capability functions
    /// take `SlangSession*`, which slang.h typedefs to this same interface.
    global_session: *mut c_void,
    /// `spGetBuildTagString()` — part of the disk-cache key, so a toolchain
    /// upgrade invalidates every cached compile.
    build_tag: String,
    find_capability: unsafe extern "C" fn(*mut c_void, *const c_char) -> i32,
    reflection_parameter_count: unsafe extern "C" fn(*mut c_void) -> c_uint,
    reflection_parameter_by_index: unsafe extern "C" fn(*mut c_void, c_uint) -> *mut c_void,
    parameter_binding_index: unsafe extern "C" fn(*mut c_void) -> c_uint,
    parameter_binding_space: unsafe extern "C" fn(*mut c_void) -> c_uint,
    variable_layout_variable: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    variable_name: unsafe extern "C" fn(*mut c_void) -> *const c_char,
}

// SAFETY: the global session is only ever dereferenced by libslang calls made
// under the `SLANG` mutex below, so cross-thread moves are serialized.
unsafe impl Send for Api {}

/// Global compiler state. A slang global session (and everything derived from
/// it) is not thread-safe; every use is serialized under this lock
/// (registration-time validation runs on the main thread, pipeline builds on
/// the render thread). A load failure is cached — it's a broken/missing
/// toolchain install, not transient.
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
            // SAFETY: signature transcribed from the pinned install's slang.h.
            *unsafe { lib.get($name) }.map_err(|e| format!("{path}: {}: {e}", $name.escape_ascii()))?
        };
    }
    let create_global_session: unsafe extern "C" fn(i64, *mut *mut c_void) -> i32 =
        sym!(b"slang_createGlobalSession");
    let get_build_tag: unsafe extern "C" fn() -> *const c_char = sym!(b"spGetBuildTagString");

    let mut global_session: *mut c_void = core::ptr::null_mut();
    // SAFETY: out-pointer is valid; SLANG_API_VERSION is 0.
    let result = unsafe { create_global_session(0, &mut global_session) };
    if result < 0 || global_session.is_null() {
        return Err(format!("{path}: slang_createGlobalSession failed ({result})"));
    }
    // SAFETY: returns a static string owned by the library.
    let build_tag = unsafe {
        std::ffi::CStr::from_ptr(get_build_tag())
            .to_string_lossy()
            .into_owned()
    };

    Ok(Api {
        find_capability: sym!(b"spFindCapability"),
        reflection_parameter_count: sym!(b"spReflection_GetParameterCount"),
        reflection_parameter_by_index: sym!(b"spReflection_GetParameterByIndex"),
        parameter_binding_index: sym!(b"spReflectionParameter_GetBindingIndex"),
        parameter_binding_space: sym!(b"spReflectionParameter_GetBindingSpace"),
        variable_layout_variable: sym!(b"spReflectionVariableLayout_GetVariable"),
        variable_name: sym!(b"spReflectionVariable_GetName"),
        global_session,
        build_tag,
        _lib: lib,
    })
}

/// Capability atoms for compute kernels tracing inline `RayQuery`s
/// (`restir_spatial`, `ray_query`) — pins the target profile to the KHR
/// ray-query SPIR-V flavor the device enables.
pub const RAY_QUERY_CAPABILITIES: &[&str] = &["spvRayQueryKHR"];

/// Compile one entry point to SPIR-V words + reflection.
///
/// `entry_source` is a module named after `entry_file`; `entry_name` is the
/// entry function inside it, whose stage comes from its `[shader("...")]`
/// attribute. The emitted `OpEntryPoint` KEEPS `entry_name` (pipeline
/// stages must pass it as `pName`). `modules` are
/// `(module_name, source)` pairs made importable to the entry (and to each
/// other, in any order). `defines` are `(key, value)` preprocessor defines
/// (`slangc -D key=value`) — the compile-out feature axes like raygen's
/// `SOLARI_SHADER_CLOCK`. `capabilities` are target capability atoms
/// (`slangc -capability x`): declaring the target's capability SET restricts
/// which SPIR-V flavor gets emitted — e.g. `spvShaderInvocationReorderNV`
/// pins SER to the NV capability/extension the device enables; without it
/// slang free-chooses and emits the EXT flavor.
pub fn compile_rt_slang(
    entry_file: &str,
    entry_source: &str,
    entry_name: &str,
    modules: &[(&str, &str)],
    defines: &[(&str, &str)],
    capabilities: &[&str],
) -> Result<CompiledShader, String> {
    let mut guard = SLANG.lock().unwrap();
    let api = match guard.get_or_insert_with(load_api) {
        Ok(api) => api,
        Err(e) => return Err(e.clone()),
    };

    let key = cache_key(
        &api.build_tag,
        entry_file,
        entry_source,
        entry_name,
        modules,
        defines,
        capabilities,
    );
    if let Some(hit) = cache_load(&key) {
        return Ok(hit);
    }
    let compiled = compile_with_session(
        api,
        entry_file,
        entry_source,
        entry_name,
        modules,
        defines,
        capabilities,
    )?;
    cache_store(&key, &compiled);
    Ok(compiled)
}

fn compile_with_session(
    api: &Api,
    entry_file: &str,
    entry_source: &str,
    entry_name: &str,
    modules: &[(&str, &str)],
    defines: &[(&str, &str)],
    capabilities: &[&str],
) -> Result<CompiledShader, String> {
    let cstr = |s: &str, what: &str| {
        CString::new(s).map_err(|_| format!("{entry_file}: interior NUL in {what}"))
    };
    let entry_module_name = cstr(entry_file.trim_end_matches(".slang"), "entry file name")?;
    let entry_path = cstr(entry_file, "entry file name")?;
    let entry_src = cstr(entry_source, "shader source")?;
    let entry = cstr(entry_name, "entry point name")?;
    let defines: Vec<(CString, CString)> = defines
        .iter()
        .map(|(k, v)| Ok((cstr(k, "define key")?, cstr(v, "define value")?)))
        .collect::<Result<_, String>>()?;
    let capabilities: Vec<CString> = capabilities
        .iter()
        .map(|c| cstr(c, "capability name"))
        .collect::<Result<_, String>>()?;

    // SAFETY: every COM call below targets a live interface with the slot
    // index counted off the pinned slang.h; all passed pointers outlive their
    // call (CStrings/descs live to end of scope); every owned reference is
    // released by a ComPtr guard.
    unsafe {
        // Target: SPIR-V with the declared capability set (the mechanism that
        // pins e.g. SER to the NV flavor — see the fn docs), plus full debug
        // info WITH the source text embedded — profilers (Nsight) attribute
        // samples to Slang source lines with zero path configuration, even
        // for hot-reloaded content that exists nowhere on disk. NonSemantic
        // debug instructions are stripped by the driver's final compile, so
        // codegen/runtime cost is unaffected; the price is SPIR-V bytes.
        let int_option = |name: i32, value: i32| CompilerOptionEntry {
            name,
            value: CompilerOptionValue {
                kind: OPTION_KIND_INT,
                int_value0: value,
                int_value1: 0,
                string_value0: core::ptr::null(),
                string_value1: core::ptr::null(),
            },
        };
        let mut target_options = vec![
            int_option(COMPILER_OPTION_DEBUG_INFORMATION, DEBUG_INFO_LEVEL_MAXIMAL),
            int_option(COMPILER_OPTION_DEBUG_INFO_INCLUDE_SOURCE, 1),
            int_option(COMPILER_OPTION_VULKAN_USE_ENTRY_POINT_NAME, 1),
        ];
        let capability_entries: Vec<CompilerOptionEntry> = capabilities
            .iter()
            .map(|cap| {
                // SLANG_CAPABILITY_UNKNOWN = 0.
                let id = (api.find_capability)(api.global_session, cap.as_ptr());
                if id <= 0 {
                    return Err(format!(
                        "{entry_file}: unknown slang capability {:?}",
                        cap.to_string_lossy()
                    ));
                }
                Ok(int_option(COMPILER_OPTION_CAPABILITY, id))
            })
            .collect::<Result<_, String>>()?;
        target_options.extend(capability_entries);
        let target = TargetDesc {
            structure_size: size_of::<TargetDesc>(),
            format: SLANG_SPIRV,
            profile: 0,
            flags: DEFAULT_TARGET_FLAGS,
            floating_point_mode: 0,
            line_directive_mode: 0,
            force_glsl_scalar_buffer_layout: false,
            compiler_option_entries: target_options.as_ptr(),
            compiler_option_entry_count: target_options.len() as u32,
        };
        let macros: Vec<PreprocessorMacroDesc> = defines
            .iter()
            .map(|(k, v)| PreprocessorMacroDesc {
                name: k.as_ptr(),
                value: v.as_ptr(),
            })
            .collect();
        let desc = SessionDesc {
            structure_size: size_of::<SessionDesc>(),
            targets: &target,
            target_count: 1,
            flags: 0,
            default_matrix_layout_mode: MATRIX_LAYOUT_ROW_MAJOR,
            search_paths: core::ptr::null(),
            search_path_count: 0,
            preprocessor_macros: macros.as_ptr(),
            preprocessor_macro_count: macros.len() as i64,
            file_system: core::ptr::null_mut(),
            enable_effect_annotations: false,
            allow_glsl_syntax: false,
            compiler_option_entries: core::ptr::null(),
            compiler_option_entry_count: 0,
            skip_spirv_validation: false,
        };
        let create_session: unsafe extern "C" fn(
            *mut c_void,
            *const SessionDesc,
            *mut *mut c_void,
        ) -> i32 =
            core::mem::transmute(com_slot(api.global_session, IGLOBAL_SESSION_CREATE_SESSION));
        let mut session_ptr: *mut c_void = core::ptr::null_mut();
        let result = create_session(api.global_session, &desc, &mut session_ptr);
        if result < 0 || session_ptr.is_null() {
            return Err(format!("{entry_file}: createSession failed ({result})"));
        }
        let session = ComPtr(session_ptr);

        let load_module: unsafe extern "C" fn(
            *mut c_void,
            *const c_char,
            *const c_char,
            *const c_char,
            *mut *mut c_void,
        ) -> *mut c_void =
            core::mem::transmute(com_slot(session.0, ISESSION_LOAD_MODULE_FROM_SOURCE_STRING));
        let diag_string = |diag: *mut c_void| -> String {
            if diag.is_null() {
                return String::new();
            }
            let s = blob_bytes(diag)
                .map(|b| String::from_utf8_lossy(b).into_owned())
                .unwrap_or_default();
            drop(ComPtr(diag));
            s
        };

        // Load the importable modules straight from source strings — imports
        // resolve against modules already loaded in the session, so iterate
        // to a fixpoint (composable modules may import each other in any
        // order). Loaded modules are owned by the session.
        let module_cstrs: Vec<(CString, CString, CString)> = modules
            .iter()
            .map(|(name, source)| {
                Ok((
                    cstr(name, "module name")?,
                    cstr(&format!("{name}.slang"), "module name")?,
                    cstr(source, "module source")?,
                ))
            })
            .collect::<Result<_, String>>()?;
        let mut pending: Vec<&(CString, CString, CString)> = module_cstrs.iter().collect();
        while !pending.is_empty() {
            let count_before = pending.len();
            let mut still_pending = Vec::new();
            let mut last_diagnostics = String::new();
            for entry in pending.drain(..) {
                let (name, path, source) = entry;
                let mut diag: *mut c_void = core::ptr::null_mut();
                let module = load_module(
                    session.0,
                    name.as_ptr(),
                    path.as_ptr(),
                    source.as_ptr(),
                    &mut diag,
                );
                let diagnostics = diag_string(diag);
                if module.is_null() {
                    last_diagnostics = diagnostics;
                    still_pending.push(entry);
                } else if !diagnostics.trim().is_empty() {
                    tracing::warn!(
                        "{}: slang diagnostics:\n{diagnostics}",
                        path.to_string_lossy()
                    );
                }
            }
            // A full round with no progress: the survivors have real errors,
            // not unresolved imports.
            if still_pending.len() == count_before {
                let (_, path, _) = still_pending[0];
                return Err(format!(
                    "{}: slang module failed to load:\n{last_diagnostics}",
                    path.to_string_lossy()
                ));
            }
            pending = still_pending;
        }

        let mut diag: *mut c_void = core::ptr::null_mut();
        let module = load_module(
            session.0,
            entry_module_name.as_ptr(),
            entry_path.as_ptr(),
            entry_src.as_ptr(),
            &mut diag,
        );
        let diagnostics = diag_string(diag);
        if module.is_null() {
            return Err(format!("{entry_file}: slang compile failed:\n{diagnostics}"));
        }
        if !diagnostics.trim().is_empty() {
            tracing::warn!("{entry_file}: slang diagnostics:\n{diagnostics}");
        }

        let find_entry_point: unsafe extern "C" fn(
            *mut c_void,
            *const c_char,
            *mut *mut c_void,
        ) -> i32 = core::mem::transmute(com_slot(module, IMODULE_FIND_ENTRY_POINT_BY_NAME));
        let mut entry_point_ptr: *mut c_void = core::ptr::null_mut();
        let result = find_entry_point(module, entry.as_ptr(), &mut entry_point_ptr);
        if result < 0 || entry_point_ptr.is_null() {
            return Err(format!(
                "{entry_file}: entry point `{entry_name}` not found (missing [shader(...)]?)"
            ));
        }
        let entry_point = ComPtr(entry_point_ptr);

        let create_composite: unsafe extern "C" fn(
            *mut c_void,
            *const *mut c_void,
            i64,
            *mut *mut c_void,
            *mut *mut c_void,
        ) -> i32 =
            core::mem::transmute(com_slot(session.0, ISESSION_CREATE_COMPOSITE_COMPONENT_TYPE));
        let components = [module, entry_point.0];
        let mut composite_ptr: *mut c_void = core::ptr::null_mut();
        let mut diag: *mut c_void = core::ptr::null_mut();
        let result = create_composite(
            session.0,
            components.as_ptr(),
            components.len() as i64,
            &mut composite_ptr,
            &mut diag,
        );
        let diagnostics = diag_string(diag);
        if result < 0 || composite_ptr.is_null() {
            return Err(format!("{entry_file}: compose failed:\n{diagnostics}"));
        }
        let composite = ComPtr(composite_ptr);

        let link: unsafe extern "C" fn(*mut c_void, *mut *mut c_void, *mut *mut c_void) -> i32 =
            core::mem::transmute(com_slot(composite.0, ICOMPONENT_TYPE_LINK));
        let mut linked_ptr: *mut c_void = core::ptr::null_mut();
        let mut diag: *mut c_void = core::ptr::null_mut();
        let result = link(composite.0, &mut linked_ptr, &mut diag);
        let diagnostics = diag_string(diag);
        if result < 0 || linked_ptr.is_null() {
            return Err(format!("{entry_file}: slang link failed:\n{diagnostics}"));
        }
        if !diagnostics.trim().is_empty() {
            tracing::warn!("{entry_file}: slang diagnostics:\n{diagnostics}");
        }
        let linked = ComPtr(linked_ptr);

        let get_code: unsafe extern "C" fn(
            *mut c_void,
            i64,
            i64,
            *mut *mut c_void,
            *mut *mut c_void,
        ) -> i32 = core::mem::transmute(com_slot(linked.0, ICOMPONENT_TYPE_GET_ENTRY_POINT_CODE));
        let mut code_ptr: *mut c_void = core::ptr::null_mut();
        let mut diag: *mut c_void = core::ptr::null_mut();
        let result = get_code(linked.0, 0, 0, &mut code_ptr, &mut diag);
        let diagnostics = diag_string(diag);
        if result < 0 || code_ptr.is_null() {
            return Err(format!("{entry_file}: code generation failed:\n{diagnostics}"));
        }
        if !diagnostics.trim().is_empty() {
            tracing::warn!("{entry_file}: slang diagnostics:\n{diagnostics}");
        }
        let code = ComPtr(code_ptr);
        let bytes = blob_bytes(code.0)
            .ok_or_else(|| format!("{entry_file}: slang produced an empty code blob"))?;
        if bytes.len() < 20 || bytes.len() % 4 != 0 {
            return Err(format!(
                "{entry_file}: slang produced truncated code ({} bytes)",
                bytes.len()
            ));
        }
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        if words[0] != 0x0723_0203 {
            return Err(format!("{entry_file}: slang output is not SPIR-V"));
        }

        // Global-parameter reflection (`ProgramLayout` == `SlangReflection`,
        // owned by `linked` — copied out before the guards drop).
        let get_layout: unsafe extern "C" fn(*mut c_void, i64, *mut *mut c_void) -> *mut c_void =
            core::mem::transmute(com_slot(linked.0, ICOMPONENT_TYPE_GET_LAYOUT));
        let mut diag: *mut c_void = core::ptr::null_mut();
        let reflection = get_layout(linked.0, 0, &mut diag);
        drop(ComPtr(diag));
        let mut bindings = Vec::new();
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

        Ok(CompiledShader {
            spirv: words,
            bindings,
        })
    }
}

/// A blob's contents (`None` when empty).
///
/// # Safety
/// `blob` must be a live `ISlangBlob`; the slice borrows from it.
unsafe fn blob_bytes<'a>(blob: *mut c_void) -> Option<&'a [u8]> {
    unsafe {
        let get_ptr: unsafe extern "C" fn(*mut c_void) -> *const c_void =
            core::mem::transmute(com_slot(blob, IBLOB_GET_BUFFER_POINTER));
        let get_size: unsafe extern "C" fn(*mut c_void) -> usize =
            core::mem::transmute(com_slot(blob, IBLOB_GET_BUFFER_SIZE));
        let ptr = get_ptr(blob);
        let size = get_size(blob);
        if ptr.is_null() || size == 0 {
            return None;
        }
        Some(std::slice::from_raw_parts(ptr.cast::<u8>(), size))
    }
}

// ── Disk cache ──────────────────────────────────────────────────────────────
// Value = the serialized `CompiledShader`; key = every compile input,
// length-prefixed (compiler build tag first, so a toolchain upgrade misses).
// The key is STORED in the cache file and byte-compared on load — the FNV
// filename hash is only a lookup hint, so collisions degrade to misses,
// never wrong shaders. All cache I/O is best-effort: any failure just
// recompiles.

const CACHE_MAGIC: u32 = u32::from_le_bytes(*b"SLN3");
/// Fingerprint of every fixed compiler option baked into a compile — part of
/// the cache key, so changing an option (not just sources) invalidates.
const CACHE_OPTIONS_TAG: &str = "g=max;src=1;epname=1";

fn cache_key(
    build_tag: &str,
    entry_file: &str,
    entry_source: &str,
    entry_name: &str,
    modules: &[(&str, &str)],
    defines: &[(&str, &str)],
    capabilities: &[&str],
) -> Vec<u8> {
    let mut key = Vec::new();
    let mut push = |s: &str| {
        key.extend_from_slice(&(s.len() as u64).to_le_bytes());
        key.extend_from_slice(s.as_bytes());
    };
    push(build_tag);
    push(CACHE_OPTIONS_TAG);
    push(entry_file);
    push(entry_source);
    push(entry_name);
    for (name, source) in modules {
        push(name);
        push(source);
    }
    for (k, v) in defines {
        push(k);
        push(v);
    }
    for cap in capabilities {
        push(cap);
    }
    key
}

fn cache_path(key: &[u8]) -> Option<PathBuf> {
    // FNV-1a 64 over the key — lookup hint only (see the section comment).
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in key {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))?;
    Some(base.join("bevy_solari/slang").join(format!("{hash:016x}.spv")))
}

fn cache_load(key: &[u8]) -> Option<CompiledShader> {
    let data = std::fs::read(cache_path(key)?).ok()?;
    let mut cursor = 0usize;
    let take = |cursor: &mut usize, n: usize| -> Option<&[u8]> {
        let slice = data.get(*cursor..*cursor + n)?;
        *cursor += n;
        Some(slice)
    };
    let read_u32 =
        |cursor: &mut usize| -> Option<u32> { Some(u32::from_le_bytes(take(cursor, 4)?.try_into().ok()?)) };
    if read_u32(&mut cursor)? != CACHE_MAGIC {
        return None;
    }
    let key_len = u64::from_le_bytes(take(&mut cursor, 8)?.try_into().ok()?) as usize;
    if take(&mut cursor, key_len)? != key {
        return None;
    }
    let binding_count = read_u32(&mut cursor)?;
    let mut bindings = Vec::with_capacity(binding_count as usize);
    for _ in 0..binding_count {
        let name_len = read_u32(&mut cursor)? as usize;
        let name = String::from_utf8(take(&mut cursor, name_len)?.to_vec()).ok()?;
        let set = read_u32(&mut cursor)?;
        let binding = read_u32(&mut cursor)?;
        bindings.push((name, set, binding));
    }
    let word_count = read_u32(&mut cursor)? as usize;
    let words_bytes = take(&mut cursor, word_count * 4)?;
    let spirv: Vec<u32> = words_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    if cursor != data.len() || spirv.first() != Some(&0x0723_0203) {
        return None;
    }
    Some(CompiledShader { spirv, bindings })
}

fn cache_store(key: &[u8], compiled: &CompiledShader) {
    let Some(path) = cache_path(key) else {
        return;
    };
    let mut data = Vec::new();
    data.extend_from_slice(&CACHE_MAGIC.to_le_bytes());
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    data.extend_from_slice(&(compiled.bindings.len() as u32).to_le_bytes());
    for (name, set, binding) in &compiled.bindings {
        data.extend_from_slice(&(name.len() as u32).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        data.extend_from_slice(&set.to_le_bytes());
        data.extend_from_slice(&binding.to_le_bytes());
    }
    data.extend_from_slice(&(compiled.spirv.len() as u32).to_le_bytes());
    for word in &compiled.spirv {
        data.extend_from_slice(&word.to_le_bytes());
    }
    // Atomic publish (concurrent processes race benignly: same content).
    let Some(parent) = path.parent() else { return };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    if std::fs::write(&tmp, &data).is_ok() {
        let _ = std::fs::rename(&tmp, &path);
    }
}
