use wasmtime::{Engine, Store, Instance};

mod wasm_ir;
mod muni_ir;
mod muni_ast;
mod parser;
mod lexer;
mod errors;

mod tests;

use wasm_ir::Emittable;



fn generate_wasm(input_path: &str, output_path: &str) -> Result<(), errors::CompileError> {
    
    let source = std::fs::read_to_string(input_path)
        .map_err(|e| errors::CompileError::LexerError(format!("Failed to read input file: {}", e)))?;
    let lexer = lexer::Lexer::new(source);
    let mut parser = parser::Parser::new(lexer);
    // parser.display_tokens();
    let ast = parser.parse_program();
    if ast.is_err() {
        eprintln!("Parsing error: {}", ast.err().unwrap());
        return Err(errors::CompileError::ParserError("parsing error".to_string()));
    }
    let ast = ast.unwrap();

    // println!("AST:");
    // ast.display();

    let modules = ast.lower()?;

    let muni_ir = modules.get(0).unwrap();
    
    // println!("Muni IR:");
    // muni_ir.display();
    
    let mut out = Vec::new();

    let wasm_ir = muni_ir.lower();
    // println!("Wasm IR:");
    // println!("{}", wasm_ir.display());
    wasm_ir.emit(&mut out);

    std::fs::write(output_path, out)
        .map_err(|e| errors::CompileError::WasmLoweringError(format!("Failed to write output file: {}", e)))?;

    Ok(())
}



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
    // find arguments passed to the program
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input_file> <output_file>", args[0]);
        return Ok(());
    }
    let input_path = &args[1];
    let output_path = &args[2];
    generate_wasm(input_path, output_path)?;

    let engine = Engine::default();
    let module = wasmtime::Module::from_file(&engine, output_path)?;
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[])?;

    let main = instance.get_typed_func::<(), i32>(&mut store, "main")?;
    let result = main.call(&mut store, ())?;
    println!("main returned: {}", result);

    Ok(())
}
