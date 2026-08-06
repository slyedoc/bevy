//! Build script that compiles the `.bsn` LALRPOP grammar (`dynamic_bsn_grammar.lalrpop`).

fn main() {
    lalrpop::process_src().unwrap();
}
