// test suite for the compiler written in simple Rust, without using any testing framework

use crate::compile_muni_to_wasm;


fn run_all_tests() {
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
                println!("✗ {}", test_name);
                failed += 1;
            }
            Err(e) => {
                println!("⚠ {} ({})", test_name, e);
                errors += 1;
            }
        }
    }
    
    println!("\n{} passed, {} failed, {} errors", passed, failed, errors);
}

fn run_test(test_dir_name: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let muni_code = std::fs::read_to_string(format!("tests/{test_dir_name}/test.mun"))
        .expect("Failed to read test.mun file");
    let expected_wasm = std::fs::read(format!("tests/{test_dir_name}/expected.wasm"))
        .expect("Failed to read expected.wasm file");
    
    let compiled_wasm = compile_muni_to_wasm(muni_code)?;
    
    if compiled_wasm == expected_wasm {
        println!("Test {test_dir_name} passed!");
        Ok(true)
    } else {
        println!("Test {test_dir_name} failed!");
        println!("Expected {} bytes, got {} bytes", expected_wasm.len(), compiled_wasm.len());
        // Show first difference
        for (i, (expected, compiled)) in expected_wasm.iter().zip(compiled_wasm.iter()).enumerate() {
            if expected != compiled {
                println!("  First difference at byte {}: expected 0x{:02x}, got 0x{:02x}", i, expected, compiled);
                break;
            }
        }
        Ok(false)
    }
}




