mod muni_ir;
mod wasm_ir;
mod muni_ast;
mod parser;
mod type_checker;
mod lexer;
mod errors;
mod run;
mod tests;


pub struct Options {
    pub show_tokens: bool,
    pub show_ast: bool,
    pub show_checked_ast: bool,
    pub show_muni_ir: bool,
    pub show_wasm_ir: bool,
    pub force: bool,
    pub libs: Vec<String>,
}




fn manage_args(args: Vec<String>) -> (Vec<String>, String, Options) {
    let mut options: Options = Options {
        show_tokens: false,
        show_ast: false,
        show_checked_ast: false,
        show_muni_ir: false,
        show_wasm_ir: false,
        force: false,
        libs: vec![],
    };

    let mut input_paths = Vec::new();
    let mut output_path = String::new();
    
    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        if arg == "-t" {
            options.show_tokens = true;
        } else if arg == "-a" {
            options.show_ast = true;
        } else if arg == "-c" {
            options.show_checked_ast = true;
        } else if arg == "-m" {
            options.show_muni_ir = true;
        } else if arg == "-w" {
            options.show_wasm_ir = true;
        } else if arg == "-f"  {
            options.force = true;
        } else if arg == "--wasi" {
            options.libs.push("src/lib/wasi-lib.mun".to_string());
        } else if arg == "--std" {
            options.libs.push("src/lib/std-lib.mun".to_string());
        } else if arg == "--test" {
            options.libs.push("src/lib/test-lib.mun".to_string());
        } else if arg == "-o" {
            if i + 1 >= args.len() {
                eprintln!("Output file not specified after -o");
                std::process::exit(1);
            }
            output_path = args[i + 1].clone();
            i = i + 1;
        } else if arg.starts_with("-") {
            eprintln!("Unknown option: {}", arg);
            std::process::exit(1);
        } else {
            input_paths.push(arg.clone());
        }
        i = i + 1;
    }

    (input_paths, output_path, options)
}


fn compile_files(input_paths: Vec<String>, output_path: String, options: Options) -> anyhow::Result<()> {
    if input_paths.is_empty() {
        eprintln!("No input files provided.");
        return Ok(());
    }

    if output_path.is_empty() {
        eprintln!("No output file specified. Use -o <output_file> to specify the output path.");
        return Ok(());
    }

    
    let wasm_bytes = run::compile_muni_to_wasm(input_paths, options);
    if let Err(errors) = &wasm_bytes {
        eprintln!("Compilation failed with the following errors:");
        for error in errors {
            eprintln!("- {}", error);
        }
        return Ok(());
    }
    std::fs::write(output_path, wasm_bytes.unwrap())?;
    
    Ok(())
}



fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() == 1 {
        println!("No input file provided. Running tests...");
        tests::run_all_tests();
        return Ok(());
    }

    if args.len() < 3 {
        eprintln!("Usage: {} <input_files> -o <output_file>", args[0]);
        return Ok(());
    }

    let (input_paths, output_path, options) = manage_args(args);

    if input_paths.is_empty() {
        eprintln!("No input files provided.");
        return Ok(());
    }

    if output_path.is_empty() {
        eprintln!("No output file specified. Use -o <output_file> to specify the output path.");
        return Ok(());
    }

    return compile_files(input_paths, output_path, options);
}

