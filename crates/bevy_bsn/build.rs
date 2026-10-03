//! Compiles the `.bsn` LALRPOP grammar.

fn main() {
    lalrpop::process_src().unwrap();
}
