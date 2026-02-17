

use crate::errors::{Position};

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
    Continue,
    While,
    For,
    Import,
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
    Dot,
    RArrow,
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

const SINGLE_CHAR_TOKENS: &[(&str, TokenKind)] = &[
    ("+", TokenKind::Operator(Operator::Plus)),
    ("*", TokenKind::Operator(Operator::Mul)),
    ("/", TokenKind::Operator(Operator::Div)),
    ("(", TokenKind::Symbol(Symbol::LParen)),
    (")", TokenKind::Symbol(Symbol::RParen)),
    ("{", TokenKind::Symbol(Symbol::LBrace)),
    ("}", TokenKind::Symbol(Symbol::RBrace)),
    (",", TokenKind::Symbol(Symbol::Comma)),
    (";", TokenKind::Symbol(Symbol::Semicolon)),
    (".", TokenKind::Operator(Operator::Dot)),
];

const KEYWORDS: &[(&str, TokenKind)] = &[
    ("if", TokenKind::Keyword(Keyword::If)),
    ("else", TokenKind::Keyword(Keyword::Else)),
    ("return", TokenKind::Keyword(Keyword::Return)),
    ("export", TokenKind::Keyword(Keyword::Export)),
    ("module", TokenKind::Keyword(Keyword::Module)),
    ("global", TokenKind::Keyword(Keyword::Global)),
    ("loop", TokenKind::Keyword(Keyword::Loop)),
    ("break", TokenKind::Keyword(Keyword::Break)),
    ("continue", TokenKind::Keyword(Keyword::Continue)),
    ("while", TokenKind::Keyword(Keyword::While)),
    ("for", TokenKind::Keyword(Keyword::For)),
    ("import", TokenKind::Keyword(Keyword::Import)),
];



impl Lexer {
    pub fn new(input: String) -> Self {
        Lexer { input, position: Position::new() }
    }

    fn is_at_end(&self) -> bool {
        self.position.index >= self.input.len()
    }

    fn peek_char(&self) -> Option<char> {
        self.peek_nth_char(0)
    }

    fn peek_nth_char(&self, n: usize) -> Option<char> {
        if self.position.index + n >= self.input.len() {
            None
        } else {
            self.input[self.position.index+n..].chars().next()
        }
    }

    fn is_in<'a>(c: &str, tokens: &'a[(&str, TokenKind)]) -> Option<&'a TokenKind> {
        tokens.iter().find(|(s, _)| s == &c).map(|(_, kind)| kind)
    }

    pub fn next_token(&mut self) -> Token {
        while !self.is_at_end() && let Some(c) = self.peek_char() && c.is_whitespace() {
            self.position.advance(c);
        }

        if self.is_at_end() {
            return Token { kind: TokenKind::EoF, position: self.position.clone() };
        }

        let character: char = self.peek_char().unwrap();

        match character {
            c if Lexer::is_in(&c.to_string(), SINGLE_CHAR_TOKENS).is_some() => {
                let token_kind: &TokenKind = Lexer::is_in(&character.to_string(), SINGLE_CHAR_TOKENS).unwrap();
                self.position.advance(character);
                Token { kind: token_kind.clone(), position: self.position.clone() }
            },
            '-' => {
                let token_pos = self.position.clone();
                self.position.advance(character);
                if self.peek_char() == Some('>') {
                    self.position.advance('>');
                    Token { kind: TokenKind::Operator(Operator::RArrow), position: token_pos }
                } else {
                    Token { kind: TokenKind::Operator(Operator::Minus), position: token_pos }
                }
            }
            '=' => {
                let token_pos = self.position.clone();
                self.position.advance(character);
                if self.peek_char() == Some('=') {
                    self.position.advance('=');
                    Token { kind: TokenKind::Operator(Operator::Eq), position: token_pos }
                } else {
                    Token { kind: TokenKind::Operator(Operator::Assign), position: token_pos }
                }
            },
            '!' => {
                self.position.advance(character);
                if self.peek_char() == Some('=') {
                    self.position.advance('=');
                    Token { kind: TokenKind::Operator(Operator::Ne), position: self.position.clone() }
                } else {
                    Token { kind: TokenKind::Operator(Operator::Not), position: self.position.clone() }
                }
            },
            '<' => {
                self.position.advance(character);
                if self.peek_char() == Some('=') {
                    self.position.advance('=');
                    Token { kind: TokenKind::Operator(Operator::Le), position: self.position.clone() }
                } else {
                    Token { kind: TokenKind::Operator(Operator::Lt), position: self.position.clone() }
                }
            },
            '>' => {
                self.position.advance(character);
                if self.peek_char() == Some('=') {
                    self.position.advance('=');
                    Token { kind: TokenKind::Operator(Operator::Ge), position: self.position.clone() }
                } else {
                    Token { kind: TokenKind::Operator(Operator::Gt), position: self.position.clone() }
                }
            },

            c if c.is_ascii_digit() => {
                self.read_number()
            },
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start: usize = self.position.index;
                while !self.is_at_end() && (self.peek_char().unwrap().is_ascii_alphanumeric() || self.peek_char().unwrap() == '_') {
                    self.position.advance(self.peek_char().unwrap());
                }
                let ident_str: &str = &self.input[start..self.position.index];
                let kind: TokenKind = match ident_str {
                    keyword if Lexer::is_in(keyword, KEYWORDS).is_some() => {
                        Lexer::is_in(keyword, KEYWORDS).unwrap().clone()
                    }
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

        while !self.is_at_end() && self.peek_char().unwrap().is_ascii_digit() {
            self.position.advance(self.peek_char().unwrap());
        }

        let kind = match self.peek_char() {
            None => TokenKind::Error,
            Some('.') => {
                self.position.advance('.');
                while !self.is_at_end() && self.peek_char().unwrap().is_ascii_digit() {
                    self.position.advance(self.peek_char().unwrap());
                }
                let num_str: &str = &self.input[start..self.position.index];
                TokenKind::Float(num_str.parse().unwrap())
            }
            Some(_) => {
                let num_str = &self.input[start..self.position.index];
                TokenKind::Integer(num_str.parse().unwrap())
            }
        };

        Token { kind, position: self.position.clone() }
    }

}
