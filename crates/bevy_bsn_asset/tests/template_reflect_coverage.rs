//! Static/dynamic parity guard: every public engine component that derives `FromTemplate` must
//! also carry `#[template(reflect)]` (and thereby a reflectable generated template), or it can be
//! spawned by the `bsn!` macro but not by a `.bsn` file.
//!
//! This is a source scan, mirroring `bevy_bsn`'s source-hygiene tests: the attribute can only be
//! added in the defining crate, so a missing annotation is not something a user can fix — it has
//! to be caught here, at the point where the gap is created.

use std::fs;
use std::path::{Path, PathBuf};

/// Public types that derive `FromTemplate` but are deliberately not `.bsn`-addressable.
/// Every entry needs a reason.
const EXCEPTIONS: &[(&str, &str)] = &[
    (
        "AtmosphereEnvironmentMap",
        "render-world internal (extracted representation), never declared in a scene",
    ),
    (
        "TemplatePatch",
        "generic over a closure; inline patches cannot be expressed in a scene file",
    ),
];

/// Crates whose sources are not scanned: the trait/derive machinery itself and the scene crates,
/// whose `FromTemplate` mentions are the derive definition, docs, and test fixtures.
const SKIPPED_CRATES: &[&str] = &["bevy_ecs", "bevy_bsn", "bevy_bsn_asset"];

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn every_public_from_template_component_has_a_reflectable_template() {
    let crates_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut violations = Vec::new();

    for entry in fs::read_dir(&crates_dir).unwrap() {
        let crate_dir = entry.unwrap().path();
        let crate_name = crate_dir.file_name().unwrap().to_string_lossy().to_string();
        let src = crate_dir.join("src");
        if !src.is_dir() || SKIPPED_CRATES.contains(&crate_name.as_str()) {
            continue;
        }
        let mut files = Vec::new();
        rust_sources(&src, &mut files);

        for file in files {
            let source = fs::read_to_string(&file).unwrap();
            let lines: Vec<&str> = source.lines().collect();
            for (index, line) in lines.iter().enumerate() {
                let trimmed = line.trim_start();
                let Some(rest) = trimmed
                    .strip_prefix("pub struct ")
                    .or_else(|| trimmed.strip_prefix("pub enum "))
                else {
                    continue;
                };
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();

                // Walk the contiguous attribute/doc block above the item.
                let mut attrs = String::new();
                let mut i = index;
                while i > 0 {
                    let above = lines[i - 1].trim_start();
                    let is_comment = above.starts_with("///") || above.starts_with("//");
                    let is_attr_ish = above.starts_with("#[")
                        || above.starts_with(")]")
                        || above.ends_with(',')
                        || above.ends_with('(')
                        || above.starts_with("derive(")
                        || above.starts_with("reflect(")
                        || above.starts_with("feature =")
                        || above.starts_with("template(");
                    if is_comment || is_attr_ish {
                        // Walk past comments, but never let their text count as attributes.
                        if !is_comment {
                            attrs.push_str(above);
                            attrs.push('\n');
                        }
                        i -= 1;
                    } else {
                        break;
                    }
                }

                let derives_from_template =
                    attrs.contains("FromTemplate") && attrs.contains("derive");
                let in_reflect_list = attrs.contains("reflect(") // avoid matching doc mentions
                    && attrs.contains("FromTemplate");
                if derives_from_template
                    && in_reflect_list
                    && !attrs.contains("template(reflect)")
                    && !EXCEPTIONS.iter().any(|(n, _)| *n == name)
                {
                    violations.push(format!("{name} ({})", file.display()));
                }
                // A `FromTemplate` derive with no reflect list at all is also a gap.
                if derives_from_template
                    && !attrs.contains("template(reflect)")
                    && !EXCEPTIONS.iter().any(|(n, _)| *n == name)
                {
                    violations.push(format!("{name} ({})", file.display()));
                }
            }
        }
    }

    violations.sort();
    violations.dedup();
    assert!(
        violations.is_empty(),
        "public components derive `FromTemplate` without `#[template(reflect)]`, so they work in \
         `bsn!` but not in `.bsn` files. Add the attribute (and `FromTemplate` to the \
         `#[reflect(...)]` list for components), or add a justified entry to EXCEPTIONS:\n{}",
        violations.join("\n")
    );
}
