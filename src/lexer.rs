#[derive(Clone)]
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


pub struct Token {
    pub kind: TokenKind,
    pub position: Position,
}

#[derive(Debug, PartialEq, Clone)]
pub enum TokenKind {
    Identifier(String),
    Integer(i64),
    Float(f64),

    Keyword(Keyword),
    Operator(Operator),
    Symbol(Symbol),
    EoF,
    Error,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Keyword {
    If,
    Else,
    Return,
    Export,
    Module,
    Global,
    Loop,
    Break,
}


#[derive(Debug, PartialEq, Clone)]
pub enum Operator {
    Plus,
    Minus,
    Mul,
    Div,
    Assign,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    Not,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Symbol {
    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    Semicolon,

}

pub struct Lexer {
    input: String,
    position: Position,
}

impl Lexer {
    pub fn new(input: String) -> Self {
        Lexer { input, position: Position::new() }
    }

    pub fn next_token(&mut self) -> Token {
        while self.position.index < self.input.len() && self.input[self.position.index..].chars().next().unwrap().is_whitespace() {
            self.position.advance(self.input[self.position.index..].chars().next().unwrap());
        }


        if self.position.index >= self.input.len() {
            return Token { kind: TokenKind::EoF, position: self.position.clone() };
        }


        let character = self.input[self.position.index..].chars().next().unwrap();



        match character {
            '+' => {
                self.position.advance(character);
                Token { kind: TokenKind::Operator(Operator::Plus), position: self.position.clone() }
            },
            '-' => {
                self.position.advance(character);
                Token { kind: TokenKind::Operator(Operator::Minus), position: self.position.clone() }
            },
            '*' => {
                self.position.advance(character);
                Token { kind: TokenKind::Operator(Operator::Mul), position: self.position.clone() }
            },
            '/' => {
                self.position.advance(character);
                Token { kind: TokenKind::Operator(Operator::Div), position: self.position.clone() }
            },
            '=' => {
                let token_pos = self.position.clone();
                self.position.advance(character);
                if self.position.index < self.input.len() && self.input[self.position.index..].chars().next().unwrap() == '=' {
                    self.position.advance('=');
                    Token { kind: TokenKind::Operator(Operator::Eq), position: token_pos }
                } else {
                    Token { kind: TokenKind::Operator(Operator::Assign), position: token_pos }
                }
            },
            '!' => {
                self.position.advance(character);
                if self.position.index < self.input.len() && self.input[self.position.index..].chars().next().unwrap() == '=' {
                    self.position.advance('=');
                    Token { kind: TokenKind::Operator(Operator::Ne), position: self.position.clone() }
                } else {
                    Token { kind: TokenKind::Operator(Operator::Not), position: self.position.clone() }
                }
            },
            '<' => {
                self.position.advance(character);
                if self.position.index < self.input.len() && self.input[self.position.index..].chars().next().unwrap() == '=' {
                    self.position.advance('=');
                    Token { kind: TokenKind::Operator(Operator::Le), position: self.position.clone() }
                } else {
                    Token { kind: TokenKind::Operator(Operator::Lt), position: self.position.clone() }
                }
            },
            '>' => {
                self.position.advance(character);
                if self.position.index < self.input.len() && self.input[self.position.index..].chars().next().unwrap() == '=' {
                    self.position.advance('=');
                    Token { kind: TokenKind::Operator(Operator::Ge), position: self.position.clone() }
                } else {
                    Token { kind: TokenKind::Operator(Operator::Gt), position: self.position.clone() }
                }
            },
            '(' => {
                self.position.advance(character);
                Token { kind: TokenKind::Symbol(Symbol::LParen), position: self.position.clone() }
            },
            ')' => {
                self.position.advance(character);
                Token { kind: TokenKind::Symbol(Symbol::RParen), position: self.position.clone() }
            },
            '{' => {
                self.position.advance(character);
                Token { kind: TokenKind::Symbol(Symbol::LBrace), position: self.position.clone() }
            },
            '}' => {
                self.position.advance(character);
                Token { kind: TokenKind::Symbol(Symbol::RBrace), position: self.position.clone() }
            },
            ',' => {
                self.position.advance(character);
                Token { kind: TokenKind::Symbol(Symbol::Comma), position: self.position.clone() }
            },
            ';' => {
                self.position.advance(character);
                Token { kind: TokenKind::Symbol(Symbol::Semicolon), position: self.position.clone() }
            },
            c if c.is_ascii_digit() => {
                self.read_number()
            },
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = self.position.index;
                while self.position.index < self.input.len() && (self.input[self.position.index..].chars().next().unwrap().is_ascii_alphanumeric() || self.input[self.position.index..].chars().next().unwrap() == '_') {
                    self.position.advance(self.input[self.position.index..].chars().next().unwrap());
                }
                let ident_str = &self.input[start..self.position.index];
                let kind = match ident_str {
                    "if" => TokenKind::Keyword(Keyword::If),
                    "else" => TokenKind::Keyword(Keyword::Else),
                    "return" => TokenKind::Keyword(Keyword::Return),
                    "break" => TokenKind::Keyword(Keyword::Break),
                    "export" => TokenKind::Keyword(Keyword::Export),
                    "module" => TokenKind::Keyword(Keyword::Module),
                    "global" => TokenKind::Keyword(Keyword::Global),
                    "loop" => TokenKind::Keyword(Keyword::Loop),
                    _ => TokenKind::Identifier(ident_str.to_string()),
                };
                Token { kind, position: self.position.clone() }
            },
            _ => {
                self.position.advance(character);
                Token { kind: TokenKind::Error, position: self.position.clone() }
            },
        }


    }


    fn read_number(&mut self) -> Token {
        let start = self.position.index;
        while self.position.index < self.input.len() && self.input[self.position.index..].chars().next().unwrap().is_ascii_digit() {
            self.position.advance(self.input[self.position.index..].chars().next().unwrap());
        }
        
        // Check for float
        let kind = 
        if self.position.index < self.input.len() && self.input[self.position.index..].chars().next().unwrap() == '.' {
            self.position.advance('.');
            while self.position.index < self.input.len() && self.input[self.position.index..].chars().next().unwrap().is_ascii_digit() {
                self.position.advance(self.input[self.position.index..].chars().next().unwrap());
            }
            let num_str = &self.input[start..self.position.index];
            TokenKind::Float(num_str.parse().unwrap())
        } else {
            let num_str = &self.input[start..self.position.index];
            TokenKind::Integer(num_str.parse().unwrap())
        };

        Token { kind, position: self.position.clone() }
    }

}
