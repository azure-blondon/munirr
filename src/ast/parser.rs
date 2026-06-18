
use crate::common::error::CompileError;
use crate::lexer::definitions::{Token, TokenKind, Operator, Keyword, Symbol};
use crate::ast::types::{Program, Module, Function, Global, HostImport, Type, TypedNode, Statement, Expression, Literal, BinOp, UnOp};

pub struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    pub fn new() -> Self {
        Parser { tokens: Vec::new(), position: 0 }
    }
    pub fn convert_tokens(&mut self, lexer: &mut crate::lexer::scan::Scanner) -> Result<(), Vec<CompileError>> {
        let mut tokens: Vec<Token> = Vec::new();
        let mut errors: Vec<CompileError> = Vec::new();
        loop {
            let token: Token = lexer.next_token();
            let is_eof: bool = matches!(token.kind, TokenKind::EoF);
            if matches!(token.kind, TokenKind::Error) {
                errors.push(CompileError::LexerError("Invalid token".to_string(), token.position.clone()));
            }
            tokens.push(token);
            if is_eof {
                break;
            }
        }
        self.tokens = tokens;
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
    
    fn is_at_end(&self) -> bool {
        self.position >= self.tokens.len()
    }

    fn nth_token(&self, n: usize) -> &Token {
        &self.tokens[self.position + n]
    }

    fn advance(&mut self) {
        if !self.is_at_end() {
            self.position += 1;
        }
    }

    fn expect(&mut self, expected: &TokenKind) -> Result<(), CompileError> {
        let token: &Token = self.nth_token(0);
        if token.kind == *expected {
            self.advance();
            Ok(())
        } else {
            Err(CompileError::ParserError(format!("Expected token {:?}, but found {:?}", expected, token.kind), token.position.clone()))
        }
    }

    pub fn parse_program(&mut self) -> Result<Program, Vec<CompileError>> {
        let mut errors: Vec<CompileError> = Vec::new();
        while self.nth_token(0).kind != TokenKind::EoF {
            match self.parse_module() {
                Ok(module) => return Ok(Program { module }),
                Err(e) => errors.push(e),
            }
        }
        Err(errors)
    }

    fn parse_module(&mut self) -> Result<Module, CompileError> {
        let mut functions: Vec<Function> = Vec::new();
        let mut globals: Vec<Global> = Vec::new();
        let mut host_imports: Vec<HostImport> = Vec::new();
        while self.nth_token(0).kind != TokenKind::EoF {
            match &self.nth_token(0).kind {
                TokenKind::Keyword(Keyword::Export) => {self.advance(); self.parse_top_level_construct(&mut functions, &mut globals, &mut host_imports, true)?},
                _ => self.parse_top_level_construct(&mut functions, &mut globals, &mut host_imports, false)?,
            }
        }
        Ok(Module { functions, globals, types: Vec::new(), host_imports })
    }

    fn parse_top_level_construct(&mut self, functions: &mut Vec<Function>, globals: &mut Vec<Global>, host_imports: &mut Vec<HostImport>, export: bool) -> Result<(), CompileError> {
        match &self.nth_token(0).kind {
            TokenKind::Identifier(_) => self.parse_function(export).map(|f| functions.push(f))?,
            TokenKind::Keyword(Keyword::Global) => self.parse_global(export).map(|g| globals.push(g))?,
            TokenKind::Keyword(Keyword::Import) => self.parse_import().map(|h| host_imports.push(h))?,
            _ => return Err(CompileError::ParserError(format!("Unexpected token {:?}", self.nth_token(0).kind), self.nth_token(0).position.clone())),
        }
        Ok(())
    }

    

    fn parse_import(&mut self) -> Result<HostImport, CompileError> {
        // import a.b(i32) -> i32;
        // import a.b(i32); (void return type)
        self.expect(&TokenKind::Keyword(Keyword::Import))?;
        let module_name = match &self.nth_token(0).kind {
            TokenKind::Identifier(name) => name.clone(),
            _ => return Err(CompileError::ParserError(format!("Expected module name, but found {:?}", self.nth_token(0).kind), self.nth_token(0).position.clone())),
        };
        self.advance();
        self.expect(&TokenKind::Operator(Operator::Dot))?;
        let func_name = match &self.nth_token(0).kind {
            TokenKind::Identifier(name) => name.clone(),
            _ => return Err(CompileError::ParserError(format!("Expected function name, but found {:?}", self.nth_token(0).kind), self.nth_token(0).position.clone())),
        };
        self.advance();
        self.expect(&TokenKind::Symbol(Symbol::LParen))?;
        let mut params: Vec<Type> = Vec::new();
        while self.nth_token(0).kind != TokenKind::Symbol(Symbol::RParen) {
            let param_type = self.parse_type()?;
            if param_type.is_none() {
                return Err(CompileError::ParserError(format!("Parameter type cannot be void"), self.nth_token(0).position.clone()));
            }
            params.push(param_type.unwrap());

            if let TokenKind::Identifier(_param_name) = &self.nth_token(0).kind {
                self.advance();
            }

            if self.nth_token(0).kind == TokenKind::Symbol(Symbol::Comma) {
                self.advance();
            }
        }
        self.expect(&TokenKind::Symbol(Symbol::RParen))?;
        if self.nth_token(0).kind == TokenKind::Symbol(Symbol::Semicolon) {
            self.advance();
            return Ok(HostImport { module: module_name, function: func_name, params, return_type: None, position: self.nth_token(0).position.clone() });
        }
        self.expect(&TokenKind::Operator(Operator::RArrow))?;
        let return_type = self.parse_type()?;
        self.expect(&TokenKind::Symbol(Symbol::Semicolon))?;
        Ok(HostImport { module: module_name, function: func_name, params, return_type, position: self.nth_token(0).position.clone() })
    }

    fn parse_global(&mut self, export: bool) -> Result<Global, CompileError> {
        self.expect(&TokenKind::Keyword(Keyword::Global))?;

        let ty = match &self.nth_token(0).kind {
            TokenKind::Identifier(_) => self.parse_type()?
                        .ok_or_else(|| CompileError::ParserError(format!("Global variable cannot have void type"), self.nth_token(0).position.clone()))?,
            _ => return Err(CompileError::ParserError(format!("Expected global variable type, but found {:?}", self.nth_token(0).kind), self.nth_token(0).position.clone())),
        };

        let name: String = match &self.nth_token(0).kind {
            TokenKind::Identifier(name) => name.clone(),
            _ => return Err(CompileError::ParserError(format!("Expected global variable name, but found {:?}", self.nth_token(0).kind), self.nth_token(0).position.clone())),
        };
        self.advance();
        self.expect(&TokenKind::Operator(Operator::Assign))?;
        let initializer = self.parse_expression()?;
        self.expect(&TokenKind::Symbol(Symbol::Semicolon))?;
        Ok(Global { name, ty, init: initializer, mutable: false, export, position: self.nth_token(0).position.clone() })
    }
    
    fn parse_function(&mut self, export: bool) -> Result<Function, CompileError> {
        if let TokenKind::Identifier(_type_name) = &self.nth_token(0).kind {
            let return_type = self.parse_type()?;
            if let TokenKind::Identifier(func_name) = &self.nth_token(0).kind {
                let func_name = func_name.clone();
                self.advance();
                self.expect(&TokenKind::Symbol(Symbol::LParen))?;
                let params = self.parse_function_params()?;
                self.expect(&TokenKind::Symbol(Symbol::RParen))?;
                self.expect(&TokenKind::Symbol(Symbol::LBrace))?;
                let body = self.parse_block()?;
                self.expect(&TokenKind::Symbol(Symbol::RBrace))?;
                Ok(Function { name: func_name, params, return_type, body, export, position: self.nth_token(0).position.clone() })
            } else {
                Err(CompileError::ParserError(format!("Expected function name, but found {:?}", self.nth_token(0).kind), self.nth_token(0).position.clone()))
            }

        } else {
            return Err(CompileError::ParserError(format!("Expected function type, but found {:?}", self.nth_token(0).kind), self.nth_token(0).position.clone()));
        }

    }


    fn parse_type(&mut self) -> Result<Option<Type>, CompileError> {
        match &self.nth_token(0).kind {
            TokenKind::Identifier(type_name) => {
                let ty: Option<Type> = match type_name.as_str() {
                    "i32" => Some(Type::I32),
                    "i64" => Some(Type::I64),
                    "f32" => Some(Type::F32),
                    "f64" => Some(Type::F64),
                    "buf" => {
                        self.advance();
                        self.expect(&TokenKind::Operator(Operator::Lt))?;
                        let inner_type = self.parse_type()?.ok_or_else(|| CompileError::ParserError(format!("Buffer type cannot be void"), self.nth_token(0).position.clone()))?;
                        self.expect(&TokenKind::Operator(Operator::Gt))?;
                        return Ok(Some(Type::Buf(Box::new(inner_type))))
                    },
                    "void" => None,
                    _ => return Err(CompileError::ParserError(format!("Unknown type '{}'", type_name), self.nth_token(0).position.clone())),
                };
                self.advance();
                Ok(ty)
            }
            _ => Err(CompileError::ParserError("Expected type".to_string(), self.nth_token(0).position.clone())),
        }
    }

    fn parse_function_params(&mut self) -> Result<Vec<(String, Type)>, CompileError> {
        let mut params: Vec<(String, Type)> = Vec::new();

        while self.nth_token(0).kind != TokenKind::Symbol(Symbol::RParen) {
            let param_type: Option<Type> = self.parse_type()?;
            if param_type.is_none() {
                return Err(CompileError::ParserError(format!("Parameter type cannot be void"), self.nth_token(0).position.clone()));
            }
            let param_type: Type = param_type.unwrap();
            let param_name: String = match &self.nth_token(0).kind {
                TokenKind::Identifier(name)  => name.clone(),
                _ => return Err(CompileError::ParserError(format!("Expected parameter name, but found {:?}", self.nth_token(0).kind), self.nth_token(0).position.clone())),
            };
            self.advance();
            params.push((param_name, param_type));
            if self.nth_token(0).kind == TokenKind::Symbol(Symbol::Comma) {
                self.advance();
            }
        }
        Ok(params)
    }

    fn parse_block(&mut self) -> Result<Vec<TypedNode>, CompileError> {
        let mut instructions: Vec<TypedNode> = Vec::new();
        while self.nth_token(0).kind != TokenKind::Symbol(Symbol::RBrace) {
            let instruction: TypedNode = self.parse_node(true)?;
            instructions.push(instruction);
        }
        Ok(instructions)
    }

    fn parse_node(&mut self, semi: bool) -> Result<TypedNode, CompileError> {
        let position = self.nth_token(0).position.clone();
        match &self.nth_token(0).kind {
            TokenKind::EoF => Err(CompileError::ParserError("Unexpected end of file".to_string(), self.nth_token(0).position.clone())),
            TokenKind::Keyword(Keyword::Loop) => {
                self.expect(&TokenKind::Keyword(Keyword::Loop))?;
                self.expect(&TokenKind::Symbol(Symbol::LBrace))?;
                let body = self.parse_block()?;
                self.expect(&TokenKind::Symbol(Symbol::RBrace))?;
                Ok(TypedNode::Statement { statement: Statement::Loop { body, position } })
            },
            TokenKind::Keyword(Keyword::If) => self.parse_if(),
            TokenKind::Keyword(Keyword::Return) => self.parse_return(),
            TokenKind::Symbol(Symbol::LBrace) => {
                self.advance();
                let block = self.parse_block()?;
                self.expect(&TokenKind::Symbol(Symbol::RBrace))?;
                Ok(TypedNode::Statement { statement: Statement::Block { body: block, position } })
            },
            TokenKind::Keyword(Keyword::Break) => {
                self.expect(&TokenKind::Keyword(Keyword::Break))?;
                if semi {
                    self.expect(&TokenKind::Symbol(Symbol::Semicolon))?;
                }
                Ok(TypedNode::Statement { statement: Statement::Break { position } })
            },
            TokenKind::Keyword(Keyword::Continue) => {
                self.expect(&TokenKind::Keyword(Keyword::Continue))?;
                if semi {
                    self.expect(&TokenKind::Symbol(Symbol::Semicolon))?;
                }
                Ok(TypedNode::Statement { statement: Statement::Continue { position } })
            },
            TokenKind::Keyword(Keyword::While) => {
                self.expect(&TokenKind::Keyword(Keyword::While))?;
                self.expect(&TokenKind::Symbol(Symbol::LParen))?;
                let condition = self.parse_expression()?;
                self.expect(&TokenKind::Symbol(Symbol::RParen))?;
                self.expect(&TokenKind::Symbol(Symbol::LBrace))?;
                let body = self.parse_block()?;
                self.expect(&TokenKind::Symbol(Symbol::RBrace))?;
                Ok(TypedNode::Statement { statement: Statement::Loop { body: vec![
                        TypedNode::Statement { statement: Statement::If {
                            condition: Box::new(condition),
                            then_body: vec![
                                TypedNode::Statement { statement: Statement::Block { body: body.clone(), position: position.clone() } },
                            ],
                            else_body: vec![
                                TypedNode::Statement { statement: Statement::Break { position: position.clone() } },
                            ],
                            position: position.clone(),
                        } },
                    ],
                    position,
                } })
            }
            TokenKind::Keyword(Keyword::For) => {
                self.expect(&TokenKind::Keyword(Keyword::For))?;
                self.expect(&TokenKind::Symbol(Symbol::LParen))?;
                let init = self.parse_node(true)?;
                let condition = self.parse_expression()?;
                self.expect(&TokenKind::Symbol(Symbol::Semicolon))?;
                let update = self.parse_node(false)?;
                self.expect(&TokenKind::Symbol(Symbol::RParen))?;
                self.expect(&TokenKind::Symbol(Symbol::LBrace))?;
                let body = self.parse_block()?;
                self.expect(&TokenKind::Symbol(Symbol::RBrace))?;

                Ok(TypedNode::Statement { statement: Statement::Block { body: vec![
                    init,
                    TypedNode::Statement { statement: Statement::Loop { body: vec![
                        TypedNode::Statement { statement: Statement::If {
                            condition: Box::new(condition),
                            then_body: vec![
                                TypedNode::Statement { statement: Statement::Block { body: body.clone(), position: position.clone() } },
                                update,
                            ],
                            else_body: vec![
                                TypedNode::Statement { statement: Statement::Break { position: position.clone() } },
                            ],
                            position: position.clone(),
                        } },

                    ],
                    position: position.clone(),
                } },
                ],
                position } })
            }
            _ => {
                // check for variable declaration : i32 x = 5;
                let saved_position: usize = self.position;
                if self.is_a_type(&self.nth_token(0).kind) {
                    if let Ok(Some(ty)) = self.parse_type() && let TokenKind::Identifier(var_name) = &self.nth_token(0).kind {
                        let var_name = var_name.clone();
                        self.advance(); 
                        let initializer = if self.nth_token(0).kind == TokenKind::Operator(Operator::Assign) {
                            self.advance();
                            Some(Box::new(self.parse_expression()?))
                        } else {
                            None
                        };
                        if semi {
                            self.expect(&TokenKind::Symbol(Symbol::Semicolon))?;
                        }
                        return Ok(TypedNode::Statement {
                            statement: Statement::VariableDeclaration { name: var_name, ty, init: initializer, position },
                        });
                    } else {
                        self.position = saved_position;
                    }
                }

                
                
                // otherwise, it's an expression statement
                let expr: TypedNode = self.parse_expression()?;
                if semi {
                    self.expect(&TokenKind::Symbol(Symbol::Semicolon))?;
                }
                Ok(TypedNode::Statement {
                    statement: Statement::Expression {
                        expression: match expr {
                            TypedNode::Expression { expression, .. } => expression,
                            _ => return Err(CompileError::ParserError("Expected expression".to_string(), position)),
                        },
                        position,
                    },
                })
            }
        }
    }

    fn parse_if(&mut self) -> Result<TypedNode, CompileError> {
        let position = self.nth_token(0).position.clone();
        self.expect(&TokenKind::Keyword(Keyword::If))?;
        let condition: Box<TypedNode> = Box::new(self.parse_expression()?);
        self.expect(&TokenKind::Symbol(Symbol::LBrace))?;
        let then_body: Vec<TypedNode> = self.parse_block()?;
        self.expect(&TokenKind::Symbol(Symbol::RBrace))?;
        
        let else_body: Vec<TypedNode> = if self.nth_token(0).kind == TokenKind::Keyword(Keyword::Else) {
            self.advance();
            self.expect(&TokenKind::Symbol(Symbol::LBrace))?;
            let body = self.parse_block()?;
            self.expect(&TokenKind::Symbol(Symbol::RBrace))?;
            body
        } else {
            Vec::new()
        };
        
        Ok(TypedNode::Statement {
            statement: Statement::If { condition, then_body, else_body, position },
        })
    }

    fn parse_return(&mut self) -> Result<TypedNode, CompileError> {
        let position = self.nth_token(0).position.clone();
        self.expect(&TokenKind::Keyword(Keyword::Return))?;
        let value = if self.nth_token(0).kind == TokenKind::Symbol(Symbol::Semicolon) {
            None
        } else {
            Some(Box::new(self.parse_expression()?))
        };
        if self.nth_token(0).kind == TokenKind::Symbol(Symbol::Semicolon) {
            self.advance();
        }
        
        Ok(TypedNode::Statement {
            statement: Statement::Return { value, position },
        })
    }

    fn parse_expression(&mut self) -> Result<TypedNode, CompileError> {
        self.parse_binary_expression(100)
    }

    fn parse_binary_expression(&mut self, max_precedence: i32) -> Result<TypedNode, CompileError> {
        let position = self.nth_token(0).position.clone();
        let mut left = self.parse_unary_expression()?;
        
        loop {
            let op_token_kind = self.nth_token(0).kind.clone();
            let precedence = self.get_operator_precedence(&op_token_kind);
            
            if precedence.is_none() || precedence.unwrap() > max_precedence {
                break;
            }
            
            let prec = precedence.unwrap();
            self.advance();
            
            let is_right_assoc = self.is_right_associative(&op_token_kind);
            let next_min = if is_right_assoc { prec } else { prec - 1 };
            
            let right = self.parse_binary_expression(next_min)?;
            
            let op = match &op_token_kind {
                TokenKind::Operator(Operator::Plus) => BinOp::Add,
                TokenKind::Operator(Operator::Minus) => BinOp::Sub,
                TokenKind::Operator(Operator::Mul) => BinOp::Mul,
                TokenKind::Operator(Operator::Div) => BinOp::Div,
                TokenKind::Operator(Operator::Mod) => BinOp::Mod,
                TokenKind::Operator(Operator::Gt) => BinOp::Gt,
                TokenKind::Operator(Operator::Lt) => BinOp::Lt,
                TokenKind::Operator(Operator::Ge) => BinOp::Ge,
                TokenKind::Operator(Operator::Le) => BinOp::Le,
                TokenKind::Operator(Operator::Eq) => BinOp::Eq,
                TokenKind::Operator(Operator::Assign) => BinOp::Assign,
                _ => return Err(CompileError::ParserError("Unknown operator".to_string(), position.clone())),
            };
            
            left = TypedNode::Expression {
                expression: Expression::BinaryOp {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                    position: position.clone(),
                },
                result_type: None,
            };
        }
        
        Ok(left)
    }

    fn parse_unary_expression(&mut self) -> Result<TypedNode, CompileError> {
        let position = self.nth_token(0).position.clone();
        match &self.nth_token(0).kind {
            TokenKind::Operator(Operator::Minus) => {
                self.advance();
                let operand = self.parse_unary_expression()?;
                Ok(TypedNode::Expression {
                    expression: Expression::UnaryOp {
                        op: UnOp::Neg,
                        operand: Box::new(operand),
                        position: position.clone(),
                    },
                    result_type: None,
                })
            },
            TokenKind::Operator(Operator::Plus) => {
                self.advance();
                self.parse_unary_expression()
            },
            TokenKind::Operator(Operator::Not) => {
                self.advance();
                let operand = self.parse_unary_expression()?;
                Ok(TypedNode::Expression {
                    expression: Expression::UnaryOp {
                        op: UnOp::Not,
                        operand: Box::new(operand),
                        position: position
                    },
                    result_type: None,
                })
            },
            _ => self.parse_postfix_expression(),
        }
    }


    fn parse_postfix_expression(&mut self) -> Result<TypedNode, CompileError> {
        let position = self.nth_token(0).position.clone();
        let mut expr = self.parse_primary()?;
        
        loop {
            match &self.nth_token(0).kind {
                TokenKind::Symbol(Symbol::LParen) => {
                    self.advance();
                    let func_name = match expr {
                        TypedNode::Expression {
                            expression: Expression::Identifier { name, position: _ },
                            ..
                        } => name,
                        _ => return Err(CompileError::ParserError("Expected function name".to_string(), position.clone())),
                    };
                    
                    let mut args = Vec::new();
                    while self.nth_token(0).kind != TokenKind::Symbol(Symbol::RParen) {
                        args.push(self.parse_expression()?);
                        if self.nth_token(0).kind == TokenKind::Symbol(Symbol::Comma) {
                            self.advance();
                        }
                    }
                    self.expect(&TokenKind::Symbol(Symbol::RParen))?;
                    
                    expr = TypedNode::Expression {
                        expression: Expression::Call {
                            function: func_name,
                            args,
                            position: position.clone(),
                        },
                        result_type: None,
                    };
                },
                TokenKind::Symbol(Symbol::LBracket) => {
                    self.advance();
                    let index = self.parse_expression()?;
                    self.expect(&TokenKind::Symbol(Symbol::RBracket))?;
                    
                    
                    expr = TypedNode::Expression {
                        expression: Expression::BufferAccess {
                            buffer:  Box::new(expr),
                            index: Box::new(index),
                            position: position.clone(),
                        },
                        result_type: None,
                    };
                },
                _ => break,
            }
        }
        
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<TypedNode, CompileError> {
        let position = self.nth_token(0).position.clone();
        match &self.nth_token(0).kind.clone() {
            TokenKind::Integer(n) => {
                let val = *n;
                self.advance();
                Ok(TypedNode::Expression {
                    expression: Expression::Literal {
                        value: Literal::Integer(val as i64),
                        position,
                    },
                    result_type: None,
                })
            },
            TokenKind::Float(f) => {
                let val = *f;
                self.advance();
                Ok(TypedNode::Expression {
                    expression: Expression::Literal {
                        value: Literal::Float(val as f64),
                        position,
                    },
                    result_type: None,
                })
            },
            TokenKind::Char(n) => {
                let val = *n;
                self.advance();
                Ok(TypedNode::Expression {
                    expression: Expression::Literal {
                        value: Literal::Character(val),
                        position,
                    },
                    result_type: None,
                })
            }
            TokenKind::String(s) => {
                let val = s.clone();
                self.advance();
                Ok(TypedNode::Expression {
                    expression: Expression::Literal {
                        value: Literal::String(val),
                        position,
                    },
                    result_type: None,
                })
            }
            TokenKind::Identifier(name) => {
                let ident = name.clone();
                self.advance();
                Ok(TypedNode::Expression {
                    expression: Expression::Identifier { name: ident, position },
                    result_type: None,
                })
            },
            TokenKind::Symbol(Symbol::LParen) => {
                self.advance();
                let expr = self.parse_expression()?;
                self.expect(&TokenKind::Symbol(Symbol::RParen))?;
                Ok(expr)
            },
            _ => Err(CompileError::ParserError(format!("Unexpected token: {:?}", self.nth_token(0).kind), position)),
        }
    }

    fn get_operator_precedence(&self, token_kind: &TokenKind) -> Option<i32> {
        match token_kind {
            TokenKind::Operator(Operator::Mul) | TokenKind::Operator(Operator::Div) | TokenKind::Operator(Operator::Mod) => Some(3),
            TokenKind::Operator(Operator::Plus) | TokenKind::Operator(Operator::Minus) => Some(4),
            TokenKind::Operator(Operator::Lt) | TokenKind::Operator(Operator::Gt)
            | TokenKind::Operator(Operator::Le) | TokenKind::Operator(Operator::Ge) => Some(9),
            TokenKind::Operator(Operator::Eq) => Some(10),
            TokenKind::Operator(Operator::Assign) => Some(14),
            _ => None,
        }
    }

    fn is_right_associative(&self, token_kind: &TokenKind) -> bool {
        matches!(token_kind, TokenKind::Operator(Operator::Assign))
    }

    fn is_a_type(&self, token_kind: &TokenKind) -> bool {
        match token_kind {
            TokenKind::Identifier(type_name) => matches!(type_name.as_str(), "i32" | "i64" | "f32" | "f64" | "void" | "buf"),
            _ => false,
        }
    }

    #[allow(unused)]
    pub fn display_tokens(&self) {
        for token in &self.tokens {
            println!("{:?} at line {}, column {}", token.kind, token.position.line, token.position.column);
        }
    }

}
