
use crate::common::position::Position;

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub position: Position,
}

#[derive(Debug, PartialEq, Clone)]
pub enum TokenKind {
    Identifier(String),
    Integer(i64),
    Float(f64),
    Char(i32),
    String(String),

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
    Mod,
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
    LBracket,
    RBracket,
}



pub const SINGLE_CHAR_TOKENS: &[(&str, TokenKind)] = &[
    ("+", TokenKind::Operator(Operator::Plus)),
    ("*", TokenKind::Operator(Operator::Mul)),
    ("/", TokenKind::Operator(Operator::Div)),
    ("%", TokenKind::Operator(Operator::Mod)),
    ("(", TokenKind::Symbol(Symbol::LParen)),
    (")", TokenKind::Symbol(Symbol::RParen)),
    ("{", TokenKind::Symbol(Symbol::LBrace)),
    ("}", TokenKind::Symbol(Symbol::RBrace)),
    ("[", TokenKind::Symbol(Symbol::LBracket)),
    ("]", TokenKind::Symbol(Symbol::RBracket)),
    (",", TokenKind::Symbol(Symbol::Comma)),
    (";", TokenKind::Symbol(Symbol::Semicolon)),
    (".", TokenKind::Operator(Operator::Dot)),
];

pub const KEYWORDS: &[(&str, TokenKind)] = &[
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