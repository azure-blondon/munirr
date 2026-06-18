use crate::Options;
use crate::errors;
use crate::run;


pub fn run_all_tests() {
    let test_dirs = std::fs::read_dir("tests").expect("Failed to read tests directory");
    let mut passed = 0;
    let mut failed = 0;
    let mut errors = 0;
    
    for entry in test_dirs {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                println!("Error reading directory entry: {}", e);
                errors += 1;
                continue;
            }
        };
        
        if !entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
            continue;
        }
        
        let test_name = entry.file_name().into_string().unwrap_or_default();
        match run_test(&test_name) {
            Ok(true) => {
                println!("✓ {}", test_name);
                passed += 1;
            }
            Ok(false) => {
                println!("X {}", test_name);
                failed += 1;
            }
            Err(e) => {
                println!("Error running test {}: {}", test_name, e.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(", "));
                errors += 1;
            }
        }
    }
    
    println!("\n{} passed, {} failed, {} errors", passed, failed, errors);
}



fn run_test(test_dir_name: &str) -> Result<bool, Vec<errors::CompileError>> {

    
    let options = Options {
        show_tokens: false,
        show_ast: false,
        show_checked_ast: false,
        show_muni_ir: false,
        show_wasm_ir: false,
        force: false,
        libs: vec!["src/lib/std-lib.mun", "src/lib/test-lib.mun", "src/lib/wasi-lib.mun"].into_iter().map(|x| x.to_string()).collect(),
    };

    let compiled_wasm = run::compile_muni_to_wasm(vec![format!("tests/{test_dir_name}/test.mun")], options)?;
    

    let result= run::run_wasm(compiled_wasm);

    if let Err(some) = result {
        return Err(vec![errors::CompileError::GenericError(some.to_string(), errors::Position::new())]);
    }
    let status = result.unwrap();

    Ok(status == 0)
}




