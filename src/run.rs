
use wasmtime::{Engine, Linker, Store};
use wasmtime_wasi::{I32Exit, WasiCtxBuilder};
use wasmtime_wasi::p1::{self, WasiP1Ctx};

use crate::errors::CompileError;
use crate::Options;
use crate::lexer::{Lexer, TokenKind};
use crate::parser::Parser;
use crate::type_checker;

use crate::wasm_ir::Emittable;

pub fn compile_muni_to_wasm(mut input_paths: Vec<String>, options: Options) -> Result<Vec<u8>, Vec<CompileError>> {
    let mut muni_code = String::new();

    input_paths.append(&mut options.libs.clone());
    
    for input_path in &input_paths {
        muni_code.push_str(&std::fs::read_to_string(input_path.clone()).unwrap_or("".to_string()));
        muni_code.push('\n');
    }

    let mut lexer = Lexer::new(muni_code);

    if options.show_tokens {
        let mut tmp_lexer = lexer.clone();
        println!("Tokens:");
        loop {
            let token = tmp_lexer.next_token();
            println!("{:?}", token);
            if token.kind == TokenKind::EoF {
                break;
            }
        }
    }

    let mut parser = Parser::new();
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


pub fn run_wasm(wasm_bytes: Vec<u8>) -> Result<i32, anyhow::Error> {
    let engine = Engine::default();
    let module = wasmtime::Module::new(&engine, wasm_bytes)?;
    
    let mut linker: Linker<WasiP1Ctx> = Linker::new(&engine);
    p1::add_to_linker_sync(&mut linker, |state| state)?;
    
    let pre = linker.instantiate_pre(&module)?;
    
    let wasi = WasiCtxBuilder::new()
        .inherit_stdio()
        .inherit_args()
        .build_p1();
    
    let mut store = Store::new(&engine, wasi);
    let instance = pre.instantiate(&mut store)?;
    
    let main = instance.get_typed_func::<(), ()>(&mut store, "_start")?;


    match main.call(&mut store, ()) {
        Ok(()) => Ok(0),
        Err(err) => {
            if let Some(exit) = err.downcast_ref::<I32Exit>() {
                Ok(exit.0)
            } else {
                Err(err)
            }
        }
    }
    
}

