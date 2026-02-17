

#[derive(Debug, Clone, Copy)]
pub struct Position {
    pub line: usize,
    pub column: usize,
    pub index: usize,
}

impl Position {
    pub fn new() -> Self {
        Position { line: 1, column: 1, index: 0 }
    }

    pub fn advance(&mut self, c: char) {
        self.index += c.len_utf8();
        if c == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
    }
}



#[derive(Debug)]
pub enum CompileError {
    LexerError(String, Position),
    ParserError(String, Position),
    TypeCheckingError(String, Position),
    IRLoweringError(String, Position),
    WasmLoweringError(String, Position),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::LexerError(msg, pos) => write!(f, "{}:{} | Lexer error: {}", pos.line, pos.column, msg),
            CompileError::ParserError(msg, pos) => write!(f, "{}:{} | Parser error: {}", pos.line, pos.column, msg),
            CompileError::TypeCheckingError(msg, pos) => write!(f, "{}:{} | Type checking error: {}", pos.line, pos.column, msg),
            CompileError::IRLoweringError(msg, pos) => write!(f, "{}:{} | IR lowering error: {}", pos.line, pos.column, msg),
            CompileError::WasmLoweringError(msg, pos) => write!(f, "{}:{} | Wasm lowering error: {}", pos.line, pos.column, msg),
        }
    }
}

impl std::error::Error for CompileError {}