use wasmtime::{Engine, Linker, Store};
use wasmtime_wasi::{WasiCtxBuilder};
mod wasm_ir;use wasmtime_wasi::p1::{self, WasiP1Ctx};

mod muni_ir;
mod muni_ast;
mod parser;
mod lexer;
mod errors;

mod tests;

use wasm_ir::Emittable;



pub fn compile_muni_to_wasm(muni_code: String) -> Result<Vec<u8>, errors::CompileError> {
    let lexer = lexer::Lexer::new(muni_code);
    let mut parser = parser::Parser::new(lexer);
    let ast = parser.parse_program()?;
    let modules: Vec<muni_ir::Module> = ast.lower()?;
    let muni_ir = modules.get(0).ok_or(errors::CompileError::IRLoweringError("No modules in program".to_string()))?;
    let mut wasm_ir = muni_ir.lower();
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




fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() == 1 {
        println!("No input file provided. Running tests...");
        tests::run_all_tests();
        return Ok(());
    }

    if args.len() < 3 {
        eprintln!("Usage: {} <input_file> <output_file>", args[0]);
        return Ok(());
    }
    let input_path = &args[1];
    let output_path = &args[2];
    let muni_code = std::fs::read_to_string(input_path)?;
    let wasm_bytes = compile_muni_to_wasm(muni_code)?;
    std::fs::write(output_path, wasm_bytes)?;
    
    Ok(())
}
