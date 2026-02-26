use wasmtime::{Engine, Linker, Store};
use wasmtime_wasi::{WasiCtxBuilder};
mod wasm_ir;use wasmtime_wasi::p1::{self, WasiP1Ctx};

mod muni_ir;
mod muni_ast;
mod parser;
mod type_checker;
mod lexer;
mod errors;

mod tests;

use wasm_ir::Emittable;

pub struct Options {
    pub show_tokens: bool,
    pub show_ast: bool,
    pub show_checked_ast: bool,
    pub show_muni_ir: bool,
    pub show_wasm_ir: bool,
    pub force: bool,
}



pub fn compile_muni_to_wasm(muni_code: String, options: Options) -> Result<Vec<u8>, Vec<errors::CompileError>> {
    
    let mut lexer = lexer::Lexer::new(muni_code);

    if options.show_tokens {
        let mut tmp_lexer = lexer.clone();
        println!("Tokens:");
        loop {
            let token = tmp_lexer.next_token();
            println!("{:?}", token);
            if token.kind == lexer::TokenKind::EoF {
                break;
            }
        }
    }

    let mut parser = parser::Parser::new();
    parser.convert_tokens(&mut lexer)?;

    
    
    let mut ast = parser.parse_program()?;

    if options.show_ast {
        println!("AST:");
        ast.display();
    }
    if !options.force {
        let mut type_checker = type_checker::TypeChecker::new();
        type_checker.check_ast(&mut ast)?;
    }
    
    if options.show_checked_ast {
        println!("AST:");
        ast.display();
    }
    
    let mut muni_ir = ast.lower()?;

    if options.show_muni_ir {
        println!("Muni IR:");
        muni_ir.display();
    }
    
    let mut wasm_ir = muni_ir.lower()?;  
    
    if options.show_wasm_ir {
        println!("WASM IR:");
        println!("{}", wasm_ir.display());
    }
    
    let mut out = Vec::new();
    
    wasm_ir.emit(&mut out);
    
    Ok(out)
}


pub fn run_wasm(wasm_bytes: Vec<u8>) -> anyhow::Result<()> {
    let engine = Engine::default();
    let module = wasmtime::Module::new(&engine, wasm_bytes)?;
    
    // Create a Linker and add imports
    let mut linker: Linker<WasiP1Ctx> = Linker::new(&engine);
    
    p1::add_to_linker_sync(&mut linker, |state| state)?;
    
    let pre = linker.instantiate_pre(&module)?;
    
    let wasi = WasiCtxBuilder::new()
        .inherit_stdio()
        .inherit_args()
        .build_p1();

    
    linker.func_wrap("env", "print", |arg: i32| {
        println!("print: {}", arg);
    })?;
    
    let mut store = Store::new(&engine, wasi);
    let instance = pre.instantiate(&mut store)?;
    
    let main = instance.get_typed_func::<(), ()>(&mut store, "_start")?;
    let result = main.call(&mut store, ())?;
    println!("main returned: {:?}", result);
    Ok(())
}


fn manage_args(args: Vec<String>) -> (Vec<String>, String, Options) {
    let mut options: Options = Options {
        show_tokens: false,
        show_ast: false,
        show_checked_ast: false,
        show_muni_ir: false,
        show_wasm_ir: false,
        force: false,
    };

    let mut input_paths = Vec::new();
    let mut output_path = String::new();
    
    for i in 1..args.len() {
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
        } else if arg == "-o" {
            if i + 1 >= args.len() {
                eprintln!("Output file not specified after -o");
                std::process::exit(1);
            }
            output_path = args[i + 1].clone();
            break;
        } else if arg.starts_with("-") {
            eprintln!("Unknown option: {}", arg);
            std::process::exit(1);
        } else {
            input_paths.push(arg.clone());
        }
    }

    (input_paths, output_path, options)
}


fn compile_files_to(input_paths: Vec<String>, output_path: String, options: Options) -> anyhow::Result<()> {
    if input_paths.is_empty() {
        eprintln!("No input files provided.");
        return Ok(());
    }

    if output_path.is_empty() {
        eprintln!("No output file specified. Use -o <output_file> to specify the output path.");
        return Ok(());
    }

    let mut muni_code = String::new(); 

    for input_path in &input_paths {
        muni_code.push_str(&std::fs::read_to_string(input_path.clone())?);
        muni_code.push('\n');
    }

    let wasm_bytes = compile_muni_to_wasm(muni_code, options);
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
    
    return compile_files_to(input_paths, output_path, options);
}
