
use crate::lexer::definitions::*;
use crate::common::position::Position;

#[derive(Debug, Clone)]
pub struct Scanner {
    input: String,
    position: Position,
}

impl Scanner {
    pub fn new(input: String) -> Self {
        Scanner { input, position: Position::new() }
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
            c if Scanner::is_in(&c.to_string(), SINGLE_CHAR_TOKENS).is_some() => {
                let token_kind: &TokenKind = Scanner::is_in(&character.to_string(), SINGLE_CHAR_TOKENS).unwrap();
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

            '\'' => {
                self.read_char_literal()
            }
            '"' => {
                self.read_string_literal()
            }

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
                    keyword if Scanner::is_in(keyword, KEYWORDS).is_some() => {
                        Scanner::is_in(keyword, KEYWORDS).unwrap().clone()
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

    fn read_char_literal(&mut self) -> Token {
        let start_pos = self.position.clone();
        self.position.advance('\'');
        if self.is_at_end() {
            return Token { kind: TokenKind::Error, position: start_pos };
        }
        let char_value = self.peek_char().unwrap();

        // handle escape sequences
        let char_value = if char_value == '\\' {
            self.position.advance('\\');
            if self.is_at_end() {
                return Token { kind: TokenKind::Error, position: start_pos };
            }
            match self.peek_char().unwrap() {
                'n' => '\n',
                't' => '\t',
                'r' => '\r',
                '\\' => '\\',
                '\'' => '\'',
                '\"' => '\"',
                other => {
                    eprintln!("Unknown escape sequence: \\{}", other);
                    return Token { kind: TokenKind::Error, position: start_pos };
                }
            }
        } else {
            char_value
        };
        self.position.advance(char_value);
        if self.is_at_end() || self.peek_char().unwrap() != '\'' {
            return Token { kind: TokenKind::Error, position: start_pos };
        }
        self.position.advance('\'');
        Token { kind: TokenKind::Char(char_value as i32), position: start_pos }
    }

    fn read_string_literal(&mut self) -> Token {
        let start_pos = self.position.clone();
        self.position.advance('"');
        let mut string_value = String::new();

        while !self.is_at_end() && self.peek_char().unwrap() != '"' {
            let c = self.peek_char().unwrap();
            if c == '\\' {
                self.position.advance('\\');
                if self.is_at_end() {
                    return Token { kind: TokenKind::Error, position: start_pos };
                }
                match self.peek_char().unwrap() {
                    'n' => string_value.push('\n'),
                    't' => string_value.push('\t'),
                    'r' => string_value.push('\r'),
                    '\\' => string_value.push('\\'),
                    '\'' => string_value.push('\''),
                    '\"' => string_value.push('\"'),
                    other => {
                        eprintln!("Unknown escape sequence: \\{}", other);
                        return Token { kind: TokenKind::Error, position: start_pos };
                    }
                }
            } else {
                string_value.push(c);
            }
            self.position.advance(c);
        }

        if self.is_at_end() || self.peek_char().unwrap() != '"' {
            return Token { kind: TokenKind::Error, position: start_pos };
        }
        self.position.advance('"');
        Token { kind: TokenKind::String(string_value), position: start_pos }
    }

}
