

#[derive(Debug)]
pub enum CompileError {
    LexerError(String),
    ParserError(String),
    IRLoweringError(String),
    WasmLoweringError(String),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::LexerError(msg) => write!(f, "Lexer error: {}", msg),
            CompileError::ParserError(msg) => write!(f, "Parser error: {}", msg),
            CompileError::IRLoweringError(msg) => write!(f, "IR lowering error: {}", msg),
            CompileError::WasmLoweringError(msg) => write!(f, "Wasm lowering error: {}", msg),
        }
    }
}

impl std::error::Error for CompileError {}