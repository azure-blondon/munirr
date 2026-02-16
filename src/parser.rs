use crate::{errors, muni_ast};
use crate::lexer::{Token, TokenKind, Operator, Keyword, Symbol};


pub struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    pub fn new(mut lexer: crate::lexer::Lexer) -> Self {
        let mut tokens: Vec<Token> = Vec::new();
        loop {
            let token: Token = lexer.next_token();
            let is_eof: bool = matches!(token.kind, TokenKind::EoF);
            tokens.push(token);
            if is_eof {
                break;
            }
        }
        Parser { tokens, position: 0 }
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

    fn expect(&mut self, expected: &TokenKind) -> Result<(), errors::CompileError> {
        let token: &Token = self.nth_token(0);
        if token.kind == *expected {
            self.advance();
            Ok(())
        } else {
            Err(errors::CompileError::ParserError(format!("Expected token {:?} at line {}, column {}, but found {:?}", expected, token.position.line, token.position.column, token.kind)))
        }
    }

    pub fn parse_program(&mut self) -> Result<muni_ast::Program, errors::CompileError> {
        let mut modules: Vec<muni_ast::Module> = Vec::new();
        while self.nth_token(0).kind != TokenKind::EoF {
            modules.push(self.parse_module()?);
        }
        Ok(muni_ast::Program { modules })
    }

    fn parse_module(&mut self) -> Result<muni_ast::Module, errors::CompileError> {
        let mut functions: Vec<muni_ast::Function> = Vec::new();
        let mut globals: Vec<muni_ast::Global> = Vec::new();
        let mut host_imports: Vec<muni_ast::HostImport> = Vec::new();
        while self.nth_token(0).kind != TokenKind::EoF {
            match &self.nth_token(0).kind {
                TokenKind::Keyword(Keyword::Export) => {self.advance(); self.parse_top_level_construct(&mut functions, &mut globals, &mut host_imports, true)?},
                _ => self.parse_top_level_construct(&mut functions, &mut globals, &mut host_imports, false)?,
            }
        }
        Ok(muni_ast::Module { functions, globals, types: Vec::new(), host_imports })
    }

    fn parse_top_level_construct(&mut self, functions: &mut Vec<muni_ast::Function>, globals: &mut Vec<muni_ast::Global>, host_imports: &mut Vec<muni_ast::HostImport>, export: bool) -> Result<(), errors::CompileError> {
        match &self.nth_token(0).kind {
            TokenKind::Identifier(_) => self.parse_function(export).map(|f| functions.push(f))?,
            TokenKind::Keyword(Keyword::Global) => self.parse_global(export).map(|g| globals.push(g))?,
            TokenKind::Keyword(Keyword::Import) => self.parse_import().map(|h| host_imports.push(h))?,
            _ => return Err(errors::CompileError::ParserError(format!("Unexpected token {:?} at line {}, column {}", self.nth_token(0).kind, self.nth_token(0).position.line, self.nth_token(0).position.column))),
        }
        Ok(())
    }

    
    fn parse_import(&mut self) -> Result<muni_ast::HostImport, errors::CompileError> {
        // import a.b(i32) -> i32;
        // import a.b(i32); (void return type)
        self.expect(&TokenKind::Keyword(Keyword::Import))?;
        let module_name = match &self.nth_token(0).kind {
            TokenKind::Identifier(name) => name.clone(),
            _ => return Err(errors::CompileError::ParserError(format!("Expected module name at line {}, column {}, but found {:?}", self.nth_token(0).position.line, self.nth_token(0).position.column, self.nth_token(0).kind))),
        };
        self.advance();
        self.expect(&TokenKind::Operator(Operator::Dot))?;
        let func_name = match &self.nth_token(0).kind {
            TokenKind::Identifier(name) => name.clone(),
            _ => return Err(errors::CompileError::ParserError(format!("Expected function name at line {}, column {}, but found {:?}", self.nth_token(0).position.line, self.nth_token(0).position.column, self.nth_token(0).kind))),
        };
        self.advance();
        self.expect(&TokenKind::Symbol(Symbol::LParen))?;
        let mut params: Vec<muni_ast::Type> = Vec::new();
        while self.nth_token(0).kind != TokenKind::Symbol(Symbol::RParen) {
            let param_type = self.parse_type()?;
            if param_type.is_none() {
                return Err(errors::CompileError::ParserError(format!("Parameter type cannot be void at line {}, column {}", self.nth_token(0).position.line, self.nth_token(0).position.column)));
            }
            params.push(param_type.unwrap());
            if self.nth_token(0).kind == TokenKind::Symbol(Symbol::Comma) {
                self.advance();
            }
        }
        self.expect(&TokenKind::Symbol(Symbol::RParen))?;
        if self.nth_token(0).kind == TokenKind::Symbol(Symbol::Semicolon) {
            self.advance();
            return Ok(muni_ast::HostImport { module: module_name, function: func_name, params, return_type: None });
        }
        self.expect(&TokenKind::Operator(Operator::RArrow))?;
        let return_type = self.parse_type()?;
        self.expect(&TokenKind::Symbol(Symbol::Semicolon))?;
        Ok(muni_ast::HostImport { module: module_name, function: func_name, params, return_type })
    }

    fn parse_global(&mut self, export: bool) -> Result<muni_ast::Global, errors::CompileError> {
        self.expect(&TokenKind::Keyword(Keyword::Global))?;

        let ty = match &self.nth_token(0).kind {
            TokenKind::Identifier(_) => self.parse_type()?
                        .ok_or_else(|| errors::CompileError::ParserError(format!("Global variable cannot have void type at line {}, column {}", self.nth_token(0).position.line, self.nth_token(0).position.column)))?,
            _ => return Err(errors::CompileError::ParserError(format!("Expected global variable type at line {}, column {}, but found {:?}", self.nth_token(0).position.line, self.nth_token(0).position.column, self.nth_token(0).kind))),
        };

        let name: String = match &self.nth_token(0).kind {
            TokenKind::Identifier(name) => name.clone(),
            _ => return Err(errors::CompileError::ParserError(format!("Expected global variable name at line {}, column {}, but found {:?}", self.nth_token(0).position.line, self.nth_token(0).position.column, self.nth_token(0).kind))),
        };
        self.advance();
        self.expect(&TokenKind::Operator(Operator::Assign))?;
        let initializer = self.parse_expression()?;
        self.expect(&TokenKind::Symbol(Symbol::Semicolon))?;
        Ok(muni_ast::Global { name, ty, init: initializer, mutable: false, export })
    }
    
    fn parse_function(&mut self, export: bool) -> Result<muni_ast::Function, errors::CompileError> {
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
                Ok(muni_ast::Function { name: func_name, params, return_type, body, export })
            } else {
                Err(errors::CompileError::ParserError(format!("Expected function name at line {}, column {}, but found {:?}", self.nth_token(0).position.line, self.nth_token(0).position.column, self.nth_token(0).kind)))
            }

        } else {
            return Err(errors::CompileError::ParserError(format!("Expected function type at line {}, column {}, but found {:?}", self.nth_token(0).position.line, self.nth_token(0).position.column, self.nth_token(0).kind)));
        }

    }


    fn parse_type(&mut self) -> Result<Option<muni_ast::Type>, errors::CompileError> {
        match &self.nth_token(0).kind {
            TokenKind::Identifier(type_name) => {
                let ty: Option<muni_ast::Type> = match type_name.as_str() {
                    "i32" => Some(muni_ast::Type::I32),
                    "i64" => Some(muni_ast::Type::I64),
                    "f32" => Some(muni_ast::Type::F32),
                    "f64" => Some(muni_ast::Type::F64),
                    "void" => None,
                    _ => return Err(errors::CompileError::ParserError(format!("Unknown type '{}'", type_name))),
                };
                self.advance();
                Ok(ty)
            }
            _ => Err(errors::CompileError::ParserError("Expected type".to_string())),
        }
    }

    fn parse_function_params(&mut self) -> Result<Vec<(String, muni_ast::Type)>, errors::CompileError> {
        let mut params: Vec<(String, muni_ast::Type)> = Vec::new();

        while self.nth_token(0).kind != TokenKind::Symbol(Symbol::RParen) {
            let param_type: Option<muni_ast::Type> = self.parse_type()?;
            if param_type.is_none() {
                return Err(errors::CompileError::ParserError(format!("Parameter type cannot be void at line {}, column {}", self.nth_token(0).position.line, self.nth_token(0).position.column)));
            }
            let param_type: muni_ast::Type = param_type.unwrap();
            let param_name: String = match &self.nth_token(0).kind {
                TokenKind::Identifier(name)  => name.clone(),
                _ => return Err(errors::CompileError::ParserError(format!("Expected parameter name at line {}, column {}, but found {:?}", self.nth_token(0).position.line, self.nth_token(0).position.column, self.nth_token(0).kind))),
            };
            self.advance();
            params.push((param_name, param_type));
            if self.nth_token(0).kind == TokenKind::Symbol(Symbol::Comma) {
                self.advance();
            }
        }
        Ok(params)
    }

    fn parse_block(&mut self) -> Result<Vec<muni_ast::TypedNode>, errors::CompileError> {
        let mut instructions: Vec<muni_ast::TypedNode> = Vec::new();
        while self.nth_token(0).kind != TokenKind::Symbol(Symbol::RBrace) {
            let instruction: muni_ast::TypedNode = self.parse_node(true)?;
            instructions.push(instruction);
        }
        Ok(instructions)
    }

    fn parse_node(&mut self, semi: bool) -> Result<muni_ast::TypedNode, errors::CompileError> {
        match &self.nth_token(0).kind {
            TokenKind::EoF => Err(errors::CompileError::ParserError("Unexpected end of file".to_string())),
            TokenKind::Keyword(Keyword::Loop) => {
                self.expect(&TokenKind::Keyword(Keyword::Loop))?;
                self.expect(&TokenKind::Symbol(Symbol::LBrace))?;
                let body = self.parse_block()?;
                self.expect(&TokenKind::Symbol(Symbol::RBrace))?;
                Ok(muni_ast::TypedNode::Statement { statement: muni_ast::Statement::Loop { body } })
            },
            TokenKind::Keyword(Keyword::If) => self.parse_if(),
            TokenKind::Keyword(Keyword::Return) => self.parse_return(),
            TokenKind::Symbol(Symbol::LBrace) => {
                self.advance();
                let block = self.parse_block()?;
                self.expect(&TokenKind::Symbol(Symbol::RBrace))?;
                Ok(muni_ast::TypedNode::Statement { statement: muni_ast::Statement::Block { body: block } })
            },
            TokenKind::Keyword(Keyword::Break) => {
                self.expect(&TokenKind::Keyword(Keyword::Break))?;
                if semi {
                    self.expect(&TokenKind::Symbol(Symbol::Semicolon))?;
                }
                Ok(muni_ast::TypedNode::Statement { statement: muni_ast::Statement::Break })
            },
            TokenKind::Keyword(Keyword::Continue) => {
                self.expect(&TokenKind::Keyword(Keyword::Continue))?;
                if semi {
                    self.expect(&TokenKind::Symbol(Symbol::Semicolon))?;
                }
                Ok(muni_ast::TypedNode::Statement { statement: muni_ast::Statement::Continue })
            },
            TokenKind::Keyword(Keyword::While) => {
                self.expect(&TokenKind::Keyword(Keyword::While))?;
                self.expect(&TokenKind::Symbol(Symbol::LParen))?;
                let condition = self.parse_expression()?;
                self.expect(&TokenKind::Symbol(Symbol::RParen))?;
                self.expect(&TokenKind::Symbol(Symbol::LBrace))?;
                let body = self.parse_block()?;
                self.expect(&TokenKind::Symbol(Symbol::RBrace))?;
                Ok(muni_ast::TypedNode::Statement { statement: muni_ast::Statement::Loop { body: vec![
                        muni_ast::TypedNode::Statement { statement: muni_ast::Statement::If {
                            condition: Box::new(condition),
                            then_body: vec![
                                muni_ast::TypedNode::Statement { statement: muni_ast::Statement::Block { body: body.clone() } },
                            ],
                            else_body: vec![
                                muni_ast::TypedNode::Statement { statement: muni_ast::Statement::Break },
                            ],
                        } },
                ] } })
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

                Ok(muni_ast::TypedNode::Statement { statement: muni_ast::Statement::Block { body: vec![
                    init,
                    muni_ast::TypedNode::Statement { statement: muni_ast::Statement::Loop { body: vec![
                        muni_ast::TypedNode::Statement { statement: muni_ast::Statement::If {
                            condition: Box::new(condition),
                            then_body: vec![
                                muni_ast::TypedNode::Statement { statement: muni_ast::Statement::Block { body: body.clone() } },
                                update,
                            ],
                            else_body: vec![
                                muni_ast::TypedNode::Statement { statement: muni_ast::Statement::Break },
                            ],
                        } },
                    ] } },
                ] } })
            }
            _ => {
                // check for variable declaration : i32 x = 5;
                let saved_position: usize = self.position;
                if let TokenKind::Identifier(_type_name) = &self.nth_token(0).kind {
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
                        return Ok(muni_ast::TypedNode::Statement {
                            statement: muni_ast::Statement::VariableDeclaration { name: var_name, ty, init: initializer },
                        });
                    } else {
                        self.position = saved_position;
                    }
                }

                let expr: muni_ast::TypedNode = self.parse_expression()?;
                if semi {
                    self.expect(&TokenKind::Symbol(Symbol::Semicolon))?;
                }
                Ok(muni_ast::TypedNode::Statement {
                    statement: muni_ast::Statement::Expression {
                        expression: match expr {
                            muni_ast::TypedNode::Expression { expression, .. } => expression,
                            _ => return Err(errors::CompileError::ParserError("Expected expression".to_string())),
                        },
                    },
                })
            }
        }
    }

    fn parse_if(&mut self) -> Result<muni_ast::TypedNode, errors::CompileError> {
        self.expect(&TokenKind::Keyword(Keyword::If))?;
        let condition: Box<muni_ast::TypedNode> = Box::new(self.parse_expression()?);
        self.expect(&TokenKind::Symbol(Symbol::LBrace))?;
        let then_body: Vec<muni_ast::TypedNode> = self.parse_block()?;
        self.expect(&TokenKind::Symbol(Symbol::RBrace))?;
        
        let else_body: Vec<muni_ast::TypedNode> = if self.nth_token(0).kind == TokenKind::Keyword(Keyword::Else) {
            self.advance();
            self.expect(&TokenKind::Symbol(Symbol::LBrace))?;
            let body = self.parse_block()?;
            self.expect(&TokenKind::Symbol(Symbol::RBrace))?;
            body
        } else {
            Vec::new()
        };
        
        Ok(muni_ast::TypedNode::Statement {
            statement: muni_ast::Statement::If { condition, then_body, else_body },
        })
    }

    fn parse_return(&mut self) -> Result<muni_ast::TypedNode, errors::CompileError> {
        self.expect(&TokenKind::Keyword(Keyword::Return))?;
        let value = if self.nth_token(0).kind == TokenKind::Symbol(Symbol::Semicolon) {
            None
        } else {
            Some(Box::new(self.parse_expression()?))
        };
        if self.nth_token(0).kind == TokenKind::Symbol(Symbol::Semicolon) {
            self.advance();
        }
        
        Ok(muni_ast::TypedNode::Statement {
            statement: muni_ast::Statement::Return { value },
        })
    }

    fn parse_expression(&mut self) -> Result<muni_ast::TypedNode, errors::CompileError> {
        self.parse_binary_expression(100)
    }

    fn parse_binary_expression(&mut self, max_precedence: i32) -> Result<muni_ast::TypedNode, errors::CompileError> {
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
                TokenKind::Operator(Operator::Plus) => muni_ast::BinOp::Add,
                TokenKind::Operator(Operator::Minus) => muni_ast::BinOp::Sub,
                TokenKind::Operator(Operator::Mul) => muni_ast::BinOp::Mul,
                TokenKind::Operator(Operator::Div) => muni_ast::BinOp::Div,
                TokenKind::Operator(Operator::Gt) => muni_ast::BinOp::Gt,
                TokenKind::Operator(Operator::Lt) => muni_ast::BinOp::Lt,
                TokenKind::Operator(Operator::Ge) => muni_ast::BinOp::Ge,
                TokenKind::Operator(Operator::Le) => muni_ast::BinOp::Le,
                TokenKind::Operator(Operator::Eq) => muni_ast::BinOp::Eq,
                TokenKind::Operator(Operator::Assign) => muni_ast::BinOp::Assign,
                _ => return Err(errors::CompileError::ParserError("Unknown operator".to_string())),
            };
            
            left = muni_ast::TypedNode::Expression {
                expression: muni_ast::Expression::BinaryOp {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                result_type: muni_ast::Type::I32, // TODO: infer type
            };
        }
        
        Ok(left)
    }

    fn parse_unary_expression(&mut self) -> Result<muni_ast::TypedNode, errors::CompileError> {
        match &self.nth_token(0).kind {
            TokenKind::Operator(Operator::Minus) => {
                self.advance();
                let operand = self.parse_unary_expression()?;
                Ok(muni_ast::TypedNode::Expression {
                    expression: muni_ast::Expression::UnaryOp {
                        op: muni_ast::UnOp::Neg,
                        operand: Box::new(operand),
                    },
                    result_type: muni_ast::Type::I32, // TODO: infer type
                })
            },
            TokenKind::Operator(Operator::Plus) => {
                self.advance();
                self.parse_unary_expression()
            },
            TokenKind::Operator(Operator::Not) => {
                self.advance();
                let operand = self.parse_unary_expression()?;
                Ok(muni_ast::TypedNode::Expression {
                    expression: muni_ast::Expression::UnaryOp {
                        op: muni_ast::UnOp::Not,
                        operand: Box::new(operand),
                    },
                    result_type: muni_ast::Type::I32, // TODO: infer type
                })
            },
            _ => self.parse_postfix_expression(),
        }
    }


    fn parse_postfix_expression(&mut self) -> Result<muni_ast::TypedNode, errors::CompileError> {
        let mut expr = self.parse_primary()?;
        
        loop {
            match &self.nth_token(0).kind {
                TokenKind::Symbol(Symbol::LParen) => {
                    self.advance();
                    let func_name = match expr {
                        muni_ast::TypedNode::Expression {
                            expression: muni_ast::Expression::Identifier { name },
                            ..
                        } => name,
                        _ => return Err(errors::CompileError::ParserError("Expected function name".to_string())),
                    };
                    
                    let mut args = Vec::new();
                    while self.nth_token(0).kind != TokenKind::Symbol(Symbol::RParen) {
                        args.push(self.parse_expression()?);
                        if self.nth_token(0).kind == TokenKind::Symbol(Symbol::Comma) {
                            self.advance();
                        }
                    }
                    self.expect(&TokenKind::Symbol(Symbol::RParen))?;
                    
                    expr = muni_ast::TypedNode::Expression {
                        expression: muni_ast::Expression::Call {
                            function: func_name,
                            args,
                        },
                        result_type: muni_ast::Type::Integer, // TODO: look up function return type
                    };
                },
                _ => break,
            }
        }
        
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<muni_ast::TypedNode, errors::CompileError> {
        match &self.nth_token(0).kind.clone() {
            TokenKind::Integer(n) => {
                let val = *n;
                self.advance();
                Ok(muni_ast::TypedNode::Expression {
                    expression: muni_ast::Expression::Literal {
                        value: muni_ast::Literal::Integer(val as i64),
                    },
                    result_type: muni_ast::Type::Integer,
                })
            },
            TokenKind::Float(f) => {
                let val = *f;
                self.advance();
                Ok(muni_ast::TypedNode::Expression {
                    expression: muni_ast::Expression::Literal {
                        value: muni_ast::Literal::Float(val as f64),
                    },
                    result_type: muni_ast::Type::Float,
                })
            },
            TokenKind::Identifier(name) => {
                let ident = name.clone();
                self.advance();
                Ok(muni_ast::TypedNode::Expression {
                    expression: muni_ast::Expression::Identifier { name: ident },
                    result_type: muni_ast::Type::Integer, // TODO: look up variable type
                })
            },
            TokenKind::Symbol(Symbol::LParen) => {
                self.advance();
                let expr = self.parse_expression()?;
                self.expect(&TokenKind::Symbol(Symbol::RParen))?;
                Ok(expr)
            },
            _ => Err(errors::CompileError::ParserError(format!("Unexpected token: {:?}", self.nth_token(0).kind))),
        }
    }

    fn get_operator_precedence(&self, token_kind: &TokenKind) -> Option<i32> {
        match token_kind {
            TokenKind::Operator(Operator::Mul) | TokenKind::Operator(Operator::Div) => Some(3),
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

    #[allow(unused)]
    pub fn display_tokens(&self) {
        for token in &self.tokens {
            println!("{:?} at line {}, column {}", token.kind, token.position.line, token.position.column);
        }
    }
}
