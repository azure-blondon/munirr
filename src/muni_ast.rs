use crate::{muni_ir, errors};
use rand;


#[derive(Debug)]
pub struct Program {
    pub module: Module,
}

#[derive(Debug)]
pub struct Module {
    pub types: Vec<TypeDef>,
    pub functions: Vec<Function>,
    pub host_imports: Vec<HostImport>,
    pub globals: Vec<Global>,
}

#[derive(Debug)]
pub struct Function {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub return_type: Option<Type>,
    pub body: Vec<TypedNode>,
    pub export: bool,
    pub position: errors::Position,
}

#[derive(Debug)]
pub struct Global {
    pub name: String,
    pub ty: Type,
    pub mutable: bool,
    pub init: TypedNode,
    pub export: bool,
    pub position: errors::Position,
}

#[derive(Debug)]
pub struct HostImport {
    pub module: String,
    pub function: String,
    pub params: Vec<Type>,
    pub return_type: Option<Type>,
    pub position: errors::Position,
}

#[derive(Debug)]
#[allow(dead_code)]
pub enum TypeDef {
    Alias { name: String, ty: Type, position: errors::Position },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Type {
    I32,
    I64,
    F32,
    F64,
}


#[derive(Debug, Clone)]
pub enum TypedNode {
    Statement { statement: Statement },
    Expression { expression: Expression, result_type: Option<Type> },
}


#[derive(Debug, Clone)]
pub enum Statement {
    If { condition: Box<TypedNode>, then_body: Vec<TypedNode>, else_body: Vec<TypedNode>, position: errors::Position },
    Return { value: Option<Box<TypedNode>>, position: errors::Position },
    Expression { expression: Expression, position: errors::Position },
    VariableDeclaration { name: String, ty: Type, init: Option<Box<TypedNode>>, position: errors::Position },
    Block { body: Vec<TypedNode>, position: errors::Position },
    Loop { body: Vec<TypedNode>, position: errors::Position },
    Break { position: errors::Position },
    Continue { position: errors::Position },
}

#[derive(Debug, Clone)]
pub enum Expression {
    BinaryOp { op: BinOp, left: Box<TypedNode>, right: Box<TypedNode>, position: errors::Position },
    UnaryOp { op: UnOp, operand: Box<TypedNode>, position: errors::Position },
    Literal { value: Literal, position: errors::Position },
    Identifier { name: String, position: errors::Position },
    Call { function: String, args: Vec<TypedNode>, position: errors::Position },
}

#[derive(Debug, Clone)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Gt,
    Lt,
    Ge,
    Le,
    Eq,
    Assign,
}

#[derive(Debug, Clone)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone)]
pub enum Literal {
    Integer(i64),
    Float(f64),
}


#[derive(Debug, Clone)]
enum BlockType {
    Loop,
    If,
    Block,
}




impl Program {
    pub fn lower(&self) -> Result<muni_ir::Module, Vec<errors::CompileError>> {
        let module = &self.module;
        let mut errors = Vec::new();
        let mut ir_functions = Vec::new();
        let mut ir_host_imports = Vec::new();
        for host_imports in &module.host_imports {
            ir_host_imports.push(muni_ir::HostImport {
                module: host_imports.module.clone(),
                function: host_imports.function.clone(),
                params: host_imports.params.iter().map(|ty| self.lower_type(ty)).collect(),
                return_type: host_imports.return_type.as_ref().map(|ty| self.lower_type(ty)),
                position: host_imports.position,
            });
        }

        for function in &module.functions {
            let ir_function = muni_ir::Function {
                name: function.name.clone(),
                params: function.params.iter().map(|(name, ty)| (name.clone(), self.lower_type(ty))).collect(),
                return_type: function.return_type.as_ref().map(|ty| self.lower_type(ty)),
                body: self.lower_instructions(&function.body, Vec::new()).unwrap_or_else(|e| {
                    errors.push(e);
                    Vec::new()
                }),
                export: function.export,
                locals: function.locals().iter().map(|(name, ty)| (name.clone(), self.lower_type(ty))).collect(),
                position: function.position,
        
            };
            ir_functions.push(ir_function);
        }
        let mut ir_globals = Vec::new();
        for global in &module.globals {
            ir_globals.push(muni_ir::Global {
                name: global.name.clone(),
                global_type: self.lower_type(&global.ty),
                mutable: global.mutable,
                init: vec![self.lower_instruction(&global.init, Vec::new()).unwrap_or_else(|e| {
                    errors.push(e);
                    muni_ir::TypedInstruction {
                        instruction: muni_ir::Instruction::Const { value: match global.ty {
                            Type::I32 => muni_ir::Value::I32(0),
                            Type::I64 => muni_ir::Value::I64(0),
                            Type::F32 => muni_ir::Value::F32(0.0),
                            Type::F64 => muni_ir::Value::F64(0.0),
                        } },
                        result_type: Some(self.lower_type(&global.ty)),
                        position: global.position,
                    }
                })],
                export: global.export,
                position: global.position,
            });
        }
        for type_def in &module.types {
            match type_def {
                TypeDef::Alias { name: _, ty: _, position: _ } => {
                    // TODO handle type aliases
                },
            }
        }
        let module = muni_ir::Module {
            local_functions: ir_functions,
            host_imports: ir_host_imports,
            globals: ir_globals,
        };
            
        if errors.is_empty() {
            Ok(module)
        } else {
            Err(errors)
        }
    }

    fn lower_type(&self, ty: &Type) -> muni_ir::Type {
        match ty {
            Type::I32 => muni_ir::Type::I32,
            Type::I64 => muni_ir::Type::I64,
            Type::F32 => muni_ir::Type::F32,
            Type::F64 => muni_ir::Type::F64,
        }
    }

    fn lower_instructions(&self, instructions: &[TypedNode], block_stack: Vec<BlockType>) -> Result<Vec<muni_ir::TypedInstruction>, errors::CompileError> {
        let mut ir_instructions = Vec::new();
        for instr in instructions {
            ir_instructions.push(self.lower_instruction(instr, block_stack.clone())?);
        }
        Ok(ir_instructions)
    }

    fn lower_instruction(&self, instruction: &TypedNode, mut block_stack: Vec<BlockType>) -> Result<muni_ir::TypedInstruction, errors::CompileError> {
        let mut instr_pos = match instruction {
            TypedNode::Statement { statement } => statement.position(),
            TypedNode::Expression { expression, result_type: _ } => expression.position(),
        };
        let instr = match instruction {
            TypedNode::Statement { statement } => match statement {
                Statement::If { condition, then_body, else_body, position } => {
                    instr_pos = *position;
                    let condition = Box::new(self.lower_instruction(condition, block_stack.clone())?);
                    block_stack.push(BlockType::If);
                    let then_body = self.lower_instructions(then_body, block_stack.clone())?;
                    let else_body = self.lower_instructions(else_body, block_stack.clone())?;
                    muni_ir::Instruction::If { condition, then_body, else_body }
                },
                Statement::Return { value, position } => {
                    instr_pos = *position;
                    let value: Option<Box<muni_ir::TypedInstruction>> = value.as_ref().and_then(|v| self.lower_instruction(v, block_stack.clone()).map(Box::new).ok());
                    muni_ir::Instruction::Return { value }
                },
                Statement::Expression { expression, position } => {
                    instr_pos = *position;
                    let expr_instr = self.lower_expression(expression)?;
                    
                    let needs_drop = match expression {
                        Expression::BinaryOp { op: _, left: _, right: _, position: _ } => true,
                        Expression::UnaryOp { op: _, operand: _, position: _ } => true,
                        Expression::Literal { value: _, position: _ } => true,
                        Expression::Identifier { name: _, position: _ } => true,
                        Expression::Call { function: function_name, args: _, position: _ }  => {
                            self.module.function_returns_value(function_name)
                        }
                    };
                    
                    if needs_drop {
                        muni_ir::Instruction::Block {
                            label: format!("expr_drop_{}", rand::random::<u64>()),
                            body: vec![
                                muni_ir::TypedInstruction {
                                    instruction: expr_instr,
                                    result_type: None,
                                    position: *position,
                                },
                                muni_ir::TypedInstruction {
                                    instruction: muni_ir::Instruction::Drop,
                                    result_type: None,
                                    position: *position,
                                }
                            ]
                        }
                    } else {
                        expr_instr
                    }
                },
                Statement::VariableDeclaration { name, ty, init, position } => {
                    instr_pos = *position;
                    let init = init.as_ref().and_then(|init| self.lower_instruction(init, block_stack.clone()).map(Box::new).ok());
                    let default = Box::new(muni_ir::TypedInstruction {
                        instruction: muni_ir::Instruction::Const { value: match ty {
                            Type::I32 => muni_ir::Value::I32(0),
                            Type::I64 => muni_ir::Value::I64(0),
                            Type::F32 => muni_ir::Value::F32(0.0),
                            Type::F64 => muni_ir::Value::F64(0.0),
                        } },
                        result_type: Some(self.lower_type(ty)),
                        position: *position,
                    });
                    muni_ir::Instruction::VarSet { name: name.clone(), value: init.unwrap_or(default) }
                },
                Statement::Block { body, position } => {
                    instr_pos = *position;
                    block_stack.push(BlockType::Block);
                    let body = self.lower_instructions(body, block_stack.clone())?;
                    muni_ir::Instruction::Block { label: format!("block_{}", rand::random::<u64>()), body }
                },
                Statement::Loop { body, position } => {
                    instr_pos = *position;
                    block_stack.push(BlockType::Loop);
                    let body = self.lower_instructions(body, block_stack.clone())?;
                    
                    let loop_label = format!("loop_{}", rand::random::<u64>());

                    muni_ir::Instruction::Block { label: loop_label.clone(), body: vec![
                        muni_ir::TypedInstruction {
                            instruction: muni_ir::Instruction::Loop { label: loop_label.clone(), body: {
                                let mut instrs: Vec<muni_ir::TypedInstruction> = body;
                                instrs.push(muni_ir::TypedInstruction {
                                    instruction: muni_ir::Instruction::Break { value: 0 },
                                    result_type: None,
                                    position: *position,
                                });
                                instrs
                            } },
                            result_type: None,
                            position: *position,
                        },
                    ] }

                },
                Statement::Break { position } => {
                    instr_pos = *position;
                    muni_ir::Instruction::Break { value: block_stack.iter().rev().enumerate().find(|(_, bt)| matches!(bt, BlockType::Loop | BlockType::Block)).map(|(idx, _)| idx).unwrap_or(0) as u32 + 1 }
                },
                Statement::Continue { position } => {
                    instr_pos = *position;
                    muni_ir::Instruction::Break { value: block_stack.iter().rev().enumerate().find(|(_, bt)| matches!(bt, BlockType::Loop | BlockType::Block)).map(|(idx, _)| idx).unwrap_or(0) as u32 }
                }
            },
            TypedNode::Expression { expression, result_type: _ } => {
                self.lower_expression(expression)?

            },
        };
        Ok(muni_ir::TypedInstruction {
            instruction: instr,
            result_type: match instruction.result_type() {
                Some(ty) => Some(self.lower_type(&ty)),
                None => None,
            },
            position: instr_pos,
        })
    
    }

    fn lower_expression(&self, expression: &Expression) -> Result<muni_ir::Instruction, errors::CompileError> {
        match expression {
            Expression::UnaryOp { op, operand, position: _ } => {
                let lowered_operand = Box::new(self.lower_instruction(operand, Vec::new())?);
                Ok(match op {
                    UnOp::Neg => muni_ir::Instruction::UnaryOp { op: muni_ir::UnOp::Neg, operand: lowered_operand },
                    UnOp::Not => muni_ir::Instruction::UnaryOp { op: muni_ir::UnOp::Not, operand: lowered_operand },
                })
            },
            Expression::BinaryOp { op, left, right, position } => {
                let lowered_left = Box::new(self.lower_instruction(left, Vec::new())?);
                let lowered_right = Box::new(self.lower_instruction(right, Vec::new())?);

                Ok(match op {
                    BinOp::Add => muni_ir::Instruction::BinaryOp { op: muni_ir::BinOp::Add, left: lowered_left, right: lowered_right },
                    BinOp::Sub => muni_ir::Instruction::BinaryOp { op: muni_ir::BinOp::Sub, left: lowered_left, right: lowered_right },
                    BinOp::Mul => muni_ir::Instruction::BinaryOp { op: muni_ir::BinOp::Mul, left: lowered_left, right: lowered_right },
                    BinOp::Div => muni_ir::Instruction::BinaryOp { op: muni_ir::BinOp::Div, left: lowered_left, right: lowered_right },
                    BinOp::Gt => muni_ir::Instruction::BinaryOp { op: muni_ir::BinOp::Gt, left: lowered_left, right: lowered_right },
                    BinOp::Lt => muni_ir::Instruction::BinaryOp { op: muni_ir::BinOp::Lt, left: lowered_left, right: lowered_right },
                    BinOp::Ge => muni_ir::Instruction::BinaryOp { op: muni_ir::BinOp::Ge, left: lowered_left, right: lowered_right },
                    BinOp::Le => muni_ir::Instruction::BinaryOp { op: muni_ir::BinOp::Le, left: lowered_left, right: lowered_right },
                    BinOp::Eq => muni_ir::Instruction::BinaryOp { op: muni_ir::BinOp::Eq, left: lowered_left, right: lowered_right },
                    BinOp::Assign => {
                        let name = match left.as_ref() {
                            TypedNode::Expression { expression: Expression::Identifier { name, position: _ }, result_type: _ } => name.clone(),
                            _ => return Err(errors::CompileError::IRLoweringError("Left-hand side of assignment must be an identifier".to_string(), *position)),
                        };
                        muni_ir::Instruction::VarSet { name: name, value: lowered_right }
                    }
                })
            },
            Expression::Literal { value, position: _ } => Ok(muni_ir::Instruction::Const { value: match value {
                Literal::Integer(i) => muni_ir::Value::I32(*i as i32), // TODO: handle different integer types
                Literal::Float(f) => muni_ir::Value::F32(*f as f32), // TODO: handle different float types
            } } ),
            Expression::Identifier { name, position: _ } => {

                Ok(muni_ir::Instruction::VarGet { name: name.clone() })
            },
            Expression::Call { function, args, position } => {
                match function.as_str() {
                    "alloc" => {
                        if args.len() != 1 {
                            return Err(errors::CompileError::IRLoweringError("alloc function must have exactly one argument".to_string(), *position));
                        }
                        let size = Box::new(self.lower_instruction(&args[0], Vec::new())?);
                        return Ok(muni_ir::Instruction::Alloc { size });
                    },
                    "load" => {
                        if args.len() != 1 {
                            return Err(errors::CompileError::IRLoweringError("load function must have exactly one argument".to_string(), *position));
                        }
                        let address = Box::new(self.lower_instruction(&args[0], Vec::new())?);
                        return Ok(muni_ir::Instruction::Load { address });
                    },
                    "store" => {
                        if args.len() != 2 {
                            return Err(errors::CompileError::IRLoweringError("store function must have exactly two arguments".to_string(), *position));
                        }
                        let address = Box::new(self.lower_instruction(&args[0], Vec::new())?);
                        let value = Box::new(self.lower_instruction(&args[1], Vec::new())?);
                        return Ok(muni_ir::Instruction::Store { address, value });
                    },
                    _ => {}
                }
                let lowered_args = args.iter().map(|arg| self.lower_instruction(arg, Vec::new())).collect::<Result<Vec<_>, _>>()?;
                Ok(muni_ir::Instruction::Call { function_name: function.clone(), args: lowered_args })
            },
        }
    }

    #[allow(unused)]
    pub fn display(&self) {
        println!("Module:");
        for type_def in &self.module.types {
            match type_def {
                TypeDef::Alias { name, ty, position } => {
                    println!("  Type alias: {} = {:?}", name, ty);
                },
            }
        }
        for global in &self.module.globals {
            println!("  Global: {}: {:?} = {:?}", global.name, global.ty, global.init);
        }
        for function in &self.module.functions {
            println!("  Function: {}({:?}) -> {:?} {{", function.name, function.params, function.return_type);
            for instr in &function.body {
                self.display_instruction(instr, 4);
            }
            println!("  }}");
        }
        
    }

    pub fn display_instruction(&self, instruction: &TypedNode, indent: usize) {
        let indent_str = " ".repeat(indent);
        match instruction {
            TypedNode::Statement { statement } => {
                println!("{}Statement:", indent_str);

                match statement {
                    Statement::If { condition, then_body, else_body, position: _ } => {
                        println!("{}If:", indent_str);
                        self.display_instruction(condition, indent + 2);
                        println!("{}Then:", indent_str);
                        for instr in then_body {
                            self.display_instruction(instr, indent + 4);
                        }
                        println!("{}Else:", indent_str);
                        for instr in else_body {
                            self.display_instruction(instr, indent + 4);
                        }
                    },
                    Statement::Return { value, position: _ } => {
                        println!("{}Return:", indent_str);
                        if let Some(value) = value {
                            self.display_instruction(value, indent + 2);
                        }
                    },
                    Statement::Expression { expression, position: _ } => {
                        println!("{}Expression:", indent_str);
                        self.display_instruction(&TypedNode::Expression { expression: expression.clone(), result_type: None }, indent + 2);
                    },
                    Statement::VariableDeclaration { name, ty, init, position: _ } => {
                        println!("{}Variable declaration: {}: {:?}", indent_str, name, ty);
                        if let Some(init) = init {
                            println!("{}Initializer:", indent_str);
                            self.display_instruction(init, indent + 2);
                        }
                    },
                    Statement::Block { body, position: _ } => {
                        println!("{}Block:", indent_str);
                        for instr in body {
                            self.display_instruction(instr, indent + 2);
                        }
                    },
                    Statement::Loop { body, position: _ } => {
                        println!("{}Loop:", indent_str);
                        for instr in body {
                            self.display_instruction(instr, indent + 2);
                        }
                    },
                    Statement::Break { position: _ } => {
                        println!("{}Break", indent_str);
                    },
                    Statement::Continue { position: _ } => {
                        println!("{}Continue", indent_str);
                    }
                }  
            },
            TypedNode::Expression { expression, result_type } => {
                println!("{}Expression: {:?} (type: {:?})", indent_str, expression, result_type);
            },
        }
    }

    
}



impl Function {


    
    pub fn locals(&self) -> Vec<(String, Type)> {
        let mut locals = Vec::new();
        for instr in &self.body {
            self.collect_locals(instr, &mut locals);
        }
        locals
    }

    pub fn collect_locals(&self, node: &TypedNode, locals: &mut Vec<(String, Type)>) {
        match node {
            TypedNode::Statement { statement } => match statement {
                Statement::If { condition, then_body, else_body, position: _ } => {
                    self.collect_locals(condition, locals);
                    for instr in then_body {
                        self.collect_locals(instr, locals);
                    }
                    for instr in else_body {
                        self.collect_locals(instr, locals);
                    }
                },
                Statement::Return { value: _, position: _ } => {},
                Statement::Expression { expression: _, position: _ } => {},
                Statement::VariableDeclaration { name, ty, init, position: _ } => {
                    locals.push((name.clone(), ty.clone()));
                    if let Some(init) = init {
                        self.collect_locals(init, locals);
                    }
                },
                Statement::Block { body, position: _ } => {
                    for instr in body {
                        self.collect_locals(instr, locals);
                    }
                },
                Statement::Loop { body, position: _ } => {
                    for instr in body {
                        self.collect_locals(instr, locals);
                    }
                },
                Statement::Break { position: _ } => {},
                Statement::Continue { position: _ } => {},
            },
            TypedNode::Expression { expression: _, result_type: _ } => {},
        }
    }
}


impl TypedNode {
    pub fn result_type(&self) -> Option<Type> {
        match self {
            TypedNode::Statement { statement: _ } => None,
            TypedNode::Expression { expression: _, result_type } => result_type.clone(),
        }
    }
}

impl Module {
    fn function_returns_value(&self, function_name: &str) -> bool {
        // Check user-defined functions
        if let Some(func) = self.get_local_function(function_name) {
            return func.return_type.is_some();
        }
        
        // Check host imports
        if let Some(host) = self.get_host_import(function_name) {
            return host.return_type.is_some();
        }
        
        // Special built-in functions
        match function_name {
            "store" => false, 
            _ => true,
        }
    }

    fn get_local_function(&self, name: &str) -> Option<&Function> {
        self.functions.iter().find(|f| f.name == name)
    }

    fn get_host_import(&self, name: &str) -> Option<&HostImport> {
        self.host_imports.iter().find(|h| h.function == name)
    }
}




impl Statement {
    pub fn position(&self) -> errors::Position {
        match self {
            Statement::If { position, .. } => *position,
            Statement::Return { position, .. } => *position,
            Statement::Expression { position, .. } => *position,
            Statement::VariableDeclaration { position, .. } => *position,
            Statement::Block { position, .. } => *position,
            Statement::Loop { position, .. } => *position,
            Statement::Break { position } => *position,
            Statement::Continue { position } => *position,
        }
    }
}

impl Expression {
    pub fn position(&self) -> errors::Position {
        match self {
            Expression::BinaryOp { position, .. } => *position,
            Expression::UnaryOp { position, .. } => *position,
            Expression::Literal { position, .. } => *position,
            Expression::Identifier { position, .. } => *position,
            Expression::Call { position, .. } => *position,
        }
    }
}