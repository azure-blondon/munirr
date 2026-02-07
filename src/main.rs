use wasmtime::{Engine, Store, Instance};

mod wasm_ir;
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
    let modules = ast.lower()?;
    let muni_ir = modules.get(0).unwrap();
    let wasm_ir = muni_ir.lower();
    let mut out = Vec::new();
    wasm_ir.emit(&mut out);
    Ok(out)
}




fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input_file> <output_file>", args[0]);
        return Ok(());
    }
    let input_path = &args[1];
    let output_path = &args[2];
    let muni_code = std::fs::read_to_string(input_path)?;
    let wasm_bytes = compile_muni_to_wasm(muni_code)?;
    std::fs::write(output_path, wasm_bytes)?;

    let engine = Engine::default();
    let module = wasmtime::Module::from_file(&engine, output_path)?;
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])?;

    let main = instance.get_typed_func::<(), i32>(&mut store, "main")?;
    let result = main.call(&mut store, ())?;
    println!("main returned: {}", result);

    Ok(())
}
