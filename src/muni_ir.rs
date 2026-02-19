use std::vec;

use crate::wasm_ir::{self, ExportDescriptor};
use crate::errors::{self, Position};

#[derive(Debug)]
pub struct Module {
    pub local_functions: Vec<Function>,
    pub globals: Vec<Global>,
    pub host_imports: Vec<HostImport>,
}


#[derive(Debug)]
#[allow(dead_code)]
pub struct Function {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub return_type: Option<Type>,
    pub body: Vec<TypedInstruction>,
    pub locals: Vec<(String, Type)>,
    pub export: bool,
    pub position: Position,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct Global {
    pub name: String,
    pub mutable: bool,
    pub global_type: Type,
    pub init: Vec<TypedInstruction>,
    pub export: bool,
    pub position: Position,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct HostImport {
    pub module: String,
    pub function: String,
    pub params: Vec<Type>,
    pub return_type: Option<Type>,
    pub position: Position,
}


#[derive(Debug, Clone)]
pub enum Type {
    I32,
    I64,
    F32,
    F64,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TypedInstruction {
    pub instruction: Instruction,
    pub result_type: Option<Type>,
    pub position: Position,
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Const { value: Value },
    UnaryOp { op: UnOp, operand: Box<TypedInstruction> },
    BinaryOp { op: BinOp, left: Box<TypedInstruction>, right: Box<TypedInstruction> },
    VarGet { name: String },
    VarSet { name: String, value: Box<TypedInstruction> },
    If { condition: Box<TypedInstruction>, then_body: Vec<TypedInstruction>, else_body: Vec<TypedInstruction> },
    Loop { label: String, body: Vec<TypedInstruction> },
    Block { label: String, body: Vec<TypedInstruction> },
    Break { value: u32 },
    Return { value: Option<Box<TypedInstruction>> },
    Call { function_name: String, args: Vec<TypedInstruction> },
    Load { address: Box<TypedInstruction> },
    Store { address: Box<TypedInstruction>, value: Box<TypedInstruction> },
    Alloc { size: Box<TypedInstruction> },
    Drop,
}

#[derive(Debug, Clone)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
}

#[derive(Debug, Clone)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone)]
pub enum Value {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}






impl Module {
    pub fn lower(&mut self) -> Result<wasm_ir::Module, Vec<errors::CompileError>> {
        let mut module = wasm_ir::Module {
            types: Vec::new(),
            globals: Vec::new(),
            memories: Vec::new(),
            functions: Vec::new(),
            host_imports: Vec::new(),
            exports: Vec::new(),
        };
        
        self.globals.push(Global {
            name: "_heap_ptr".to_string(),
            mutable: true,
            global_type: Type::I32,
            init: vec![TypedInstruction {
                instruction: Instruction::Const { value: Value::I32(1024) },
                result_type: Some(Type::I32),
                position: Position { line: 0, column: 0, index: 0 },
            }],
            export: false,
            position: Position { line: 0, column: 0, index: 0 },
        });

        self.globals.push(Global {
            name: "_temp_ptr".to_string(),
            mutable: true,
            global_type: Type::I32,
            init: vec![TypedInstruction {
                instruction: Instruction::Const { value: Value::I32(0) },
                result_type: Some(Type::I32),
                position: Position { line: 0, column: 0, index: 0 },
            }],
            export: false,
            position: Position { line: 0, column: 0, index: 0 },
        });
        

        let mut function_indices: Vec<String> = Vec::new();

        for host_import in &self.host_imports {
            function_indices.push(host_import.function.clone());
        }


        for function in &self.local_functions {
            function_indices.push(function.name.clone());
        }


        for global in &self.globals {

            let mut init = Vec::new();

            for instr in &global.init {
                init.extend(self.lower_instruction(instr, &function_indices, None, &mut Vec::new(), &mut 0)?);
            }

            module.globals.push(wasm_ir::Global {
                global_type: match global.global_type {
                    Type::I32 => wasm_ir::ValueType::NumType { num_type: wasm_ir::NumType::I32 },
                    Type::I64 => wasm_ir::ValueType::NumType { num_type: wasm_ir::NumType::I64 },
                    Type::F32 => wasm_ir::ValueType::NumType { num_type: wasm_ir::NumType::F32 },
                    Type::F64 => wasm_ir::ValueType::NumType { num_type: wasm_ir::NumType::F64 },
                },
                mutable: match global.mutable {
                    true => wasm_ir::Mutability::Mutable,
                    false => wasm_ir::Mutability::Immutable,
                },
                init,
            });
            if global.export {
                module.exports.push(wasm_ir::Export {
                    name: global.name.clone(),
                    descriptor: ExportDescriptor::GlobalIndex(module.globals.len() as u32 - 1),
                });
            }
        }


        for (idx, function) in function_indices.iter().enumerate() {
            let function = match self.get_local_function(function) {
                Some(func) => func,
                None => match self.get_host_import(function) {
                    Some(host) => {
                        module.host_imports.push(wasm_ir::HostImport {
                            module: host.module.clone(),
                            function: host.function.clone(),
                            type_index: self.find_or_create_type_index(&mut module.types, &wasm_ir::FunctionType {
                                inputs: host.params.iter().map(|ty| match self.lower_type(ty) {
                                    wasm_ir::Type::ValueType { value_type } => value_type,
                                }).collect(),
                                outputs: match &host.return_type {
                                    None => vec![],
                                    Some(ty) => vec![match self.lower_type(ty) {
                                        wasm_ir::Type::ValueType { value_type } => value_type,
                                    }],
                                },
                            }),
                        });
                        continue;
                    },
                    None => return Err(vec![
                        errors::CompileError::IRLoweringError(format!("Function not found: {}", function), Position { line: 0, column: 0, index: 0 })
                    ]),
                }
            };

            let func_type = wasm_ir::FunctionType {
                inputs: function.params.iter().map(|(_, ty)| match self.lower_type(ty) {
                    wasm_ir::Type::ValueType { value_type } => value_type,
                }).collect(),
                outputs: match &function.return_type {
                    None => vec![],
                    Some(ty) => vec![match self.lower_type(ty) {
                        wasm_ir::Type::ValueType { value_type } => value_type,
                    }],
                },
            };
            let type_index = self.find_or_create_type_index(&mut module.types, &func_type);

            
            let mut body = Vec::new();
            for instr in &function.body {
                body.extend(self.lower_instruction(instr, &function_indices, Some(idx), &mut Vec::new(), &mut 0)?);
            }
            
            module.functions.push(wasm_ir::Function {
                type_index,
                locals: function.locals.iter().map(|(_, ty)| match self.lower_type(ty) {
                    wasm_ir::Type::ValueType { value_type } => value_type,
                }).collect(),
                body: body,
            });
            
            if function.export {
                module.exports.push(wasm_ir::Export {
                    name: function.name.clone(),
                    descriptor: ExportDescriptor::FunctionIndex(self.get_function_local_index(&function.name).unwrap() as u32),
                });
            }
        }


        // TODO handle memory
        module.memories.push(wasm_ir::Memory {
            min_pages: 1,
            max_pages: None,
        });

        module.exports.push(wasm_ir::Export {
            name: "memory".to_string(),
            descriptor: ExportDescriptor::MemoryIndex(0),
        });
        
        Ok(module)
    }


    fn lower_instruction(
        &self,
        instruction: &TypedInstruction,
        function_indices: &Vec<String>,
        current_function_index: Option<usize>,
        label_stack: &mut Vec<(String, u32)>,
        next_label_id: &mut u32,
    ) -> Result<Vec<wasm_ir::Instruction>, Vec<errors::CompileError>> {
        match &instruction.instruction {
            Instruction::Const { value } => Ok(vec![self.lower_value(&value)]),
            Instruction::UnaryOp { op, operand } => {
                let mut instrs = self.lower_instruction(operand, function_indices, current_function_index, label_stack, next_label_id)?;
                let operand_type = operand.result_type.as_ref();
                if operand_type.is_none() {
                    return Err(vec![
                        errors::CompileError::IRLoweringError("Unary operation operand must have a type".to_string(), instruction.position)
                    ]);
                }
                let operand_type = operand_type.unwrap();
                let unop_instrs = self.find_unop(op, operand_type).ok_or(vec![
                    errors::CompileError::IRLoweringError(format!("Unsupported unary operation: {:?} with type {:?}", op, instruction.result_type), instruction.position)
                ])?;
                instrs.extend(unop_instrs);
                Ok(instrs)
                
            }
            Instruction::BinaryOp { op, left, right } => {
                let mut instrs = self.lower_instruction(left, function_indices, current_function_index, label_stack, next_label_id)?;
                instrs.extend(self.lower_instruction(right, function_indices, current_function_index, label_stack, next_label_id)?);

                let result_type = instruction.result_type.as_ref();
                if result_type.is_none() {
                    return Err(vec![
                        errors::CompileError::IRLoweringError("Binary operation must have a result type".to_string(), instruction.position)
                    ]);
                }
                let left_type = left.result_type.as_ref();
                if left_type.is_none() {
                    return Err(vec![
                        errors::CompileError::IRLoweringError("Binary operation left operand must have a type".to_string(), instruction.position)
                    ]);
                }
                let left_type = left_type.unwrap();

                if let Some(binop_instr) = self.find_binop(op, left_type) {
                    instrs.push(binop_instr);
                } else {
                    return Err(vec![
                        errors::CompileError::IRLoweringError(format!("Unsupported binary operation: {:?} with type {:?}", op, instruction.result_type), instruction.position)
                    ]);
                }
                Ok(instrs)
            },
            Instruction::Load { address } => {
                let mut instrs = self.lower_instruction(address, function_indices, current_function_index, label_stack, next_label_id)?;
                if let Some(load_instr) = self.find_memory_instruction(&instruction.instruction, instruction.result_type.as_ref().expect("Load operation must have a result type")) {
                    instrs.push(load_instr);
                } else {
                    return Err(vec![
                        errors::CompileError::IRLoweringError(format!("Unsupported load operation with type {:?}", instruction.result_type), instruction.position)
                    ]);
                }
                Ok(instrs)
            },
            Instruction::Store { address, value } => {
                let mut instrs = self.lower_instruction(address, function_indices, current_function_index, label_stack, next_label_id)?;
                instrs.extend(self.lower_instruction(value, function_indices, current_function_index, label_stack, next_label_id)?);
                
                let value_type = value.result_type.as_ref();
                if value_type.is_none() {
                    return Err(vec![
                        errors::CompileError::IRLoweringError("Store operation value must have a type".to_string(), instruction.position)
                    ]);
                }
                let value_type = value_type.unwrap();
                
                if let Some(store_instr) = self.find_memory_instruction(&instruction.instruction, value_type) {
                    instrs.push(store_instr);
                } else {
                    return Err(vec![
                        errors::CompileError::IRLoweringError(format!("Unsupported store operation with type {:?}", instruction.result_type), instruction.position)
                    ]);
                }
                Ok(instrs)
            },

            Instruction::Alloc { size } => {
                let mut instrs = Vec::new();
                
                // Evaluate size once and store it
                let size_instrs = self.lower_instruction(size, function_indices, current_function_index, label_stack, next_label_id)?;
                
                // ptr = load _heap_ptr
                instrs.push(wasm_ir::Instruction::GlobalGet { id: self.get_global_index("_heap_ptr")? });
                instrs.push(wasm_ir::Instruction::GlobalSet { id: self.get_global_index("_temp_ptr")? });
                
                // Store length at ptr: *ptr = size
                instrs.push(wasm_ir::Instruction::GlobalGet { id: self.get_global_index("_temp_ptr")? });
                instrs.extend(size_instrs.clone());  // Use the cached size
                instrs.push(wasm_ir::Instruction::I32Store { align: 1, offset: 0 });
                
                // Increment _heap_ptr by (4 + size)
                instrs.push(wasm_ir::Instruction::GlobalGet { id: self.get_global_index("_heap_ptr")? });
                instrs.push(wasm_ir::Instruction::I32Const { value: 4 });
                instrs.extend(size_instrs.clone());  // Use the cached size again
                instrs.push(wasm_ir::Instruction::I32Add);
                instrs.push(wasm_ir::Instruction::I32Add);
                instrs.push(wasm_ir::Instruction::GlobalSet { id: self.get_global_index("_heap_ptr")? });
                
                // Return ptr + 4 (data start, skipping length)
                instrs.push(wasm_ir::Instruction::GlobalGet { id: self.get_global_index("_temp_ptr")? });
                instrs.push(wasm_ir::Instruction::I32Const { value: 4 });
                instrs.push(wasm_ir::Instruction::I32Add);
                
                Ok(instrs)
            }

            Instruction::VarGet { name } => {
                if let Some(func_idx) = current_function_index {
                    let local_func_idx = func_idx - self.host_imports.len();
                    if self.get_locals_names(local_func_idx).iter().any(|local_name| local_name == name) {
                        return Ok(vec![wasm_ir::Instruction::LocalGet { id: self.get_local_index(local_func_idx, name) }]);
                    }
                }

                Ok(vec![wasm_ir::Instruction::GlobalGet { id: self.get_global_index(name)? }])
            },
            Instruction::VarSet { name, value } => {
                if let Some(func_idx) = current_function_index {
                    let local_func_idx = func_idx - self.host_imports.len();
                    if self.local_functions[local_func_idx].params.iter().any(|(param_name, _)| param_name == name) || self.local_functions[local_func_idx].locals.iter().any(|(local_name, _)| local_name == name) {
                        let mut instrs = self.lower_instruction(value, function_indices, current_function_index, label_stack, next_label_id)?;
                        instrs.push(wasm_ir::Instruction::LocalSet { id: self.get_local_index(local_func_idx, name) });
                        return Ok(instrs);
                    }
                }
                Ok(vec![wasm_ir::Instruction::GlobalSet { id: self.get_global_index(name)? }])
            },
            Instruction::If { condition, then_body, else_body } => {

                let mut instrs = self.lower_instruction(condition, function_indices, current_function_index, label_stack, next_label_id)?;

                let mut then_body_instrs = Vec::new();
                for instr in then_body {
                    then_body_instrs.extend(self.lower_instruction(instr, function_indices, current_function_index, label_stack, next_label_id)?);
                }
                let mut else_body_instrs = Vec::new();
                for instr in else_body {
                    else_body_instrs.extend(self.lower_instruction(instr, function_indices, current_function_index, label_stack, next_label_id)?);
                }


                instrs.push(wasm_ir::Instruction::If {
                    block_type: wasm_ir::BlockType::ValueTypes { value_types: Vec::new() },
                    then_body: then_body_instrs,
                    else_body: else_body_instrs,
                });
                
                let both_branches_return = self.does_it_return(then_body) && self.does_it_return(else_body);

                if both_branches_return {
                    instrs.push(wasm_ir::Instruction::Unreachable);
                }

                Ok(instrs)
            },
            Instruction::Block { label, body } => {
                let mut instrs = Vec::new();
                let label_id = *next_label_id;
                *next_label_id += 1;
                label_stack.push((label.clone(), label_id));

                let mut body_instrs = Vec::new();
                for instr in body {
                    body_instrs.extend(self.lower_instruction(instr, function_indices, current_function_index, label_stack, next_label_id)?);
                }
                label_stack.pop();
                instrs.push(wasm_ir::Instruction::Block {
                    block_type: wasm_ir::BlockType::ValueTypes { value_types: Vec::new() },
                    body: body_instrs,
                });
                if self.does_it_return(body) {
                    instrs.push(wasm_ir::Instruction::Unreachable);
                }
                Ok(instrs)
            },
            Instruction::Loop { label, body } => {
                let mut instrs = Vec::new();
                let label_id = *next_label_id;
                *next_label_id += 1;
                label_stack.push((label.clone(), label_id));
                let mut body_instrs = Vec::new();
                for instr in body {
                    body_instrs.extend(self.lower_instruction(instr, function_indices, current_function_index, label_stack, next_label_id)?);
                }
                label_stack.pop();
                instrs.push(wasm_ir::Instruction::Loop {
                    block_type: wasm_ir::BlockType::ValueTypes { value_types: Vec::new() },
                    body: body_instrs,
                });
                if self.does_it_return(body) {
                    instrs.push(wasm_ir::Instruction::Unreachable);
                }
                Ok(instrs)
            },
            Instruction::Break { value } => {
                let mut instrs = Vec::new();
                instrs.push(wasm_ir::Instruction::Br { label_index: *value });
                Ok(instrs)
            },
            Instruction::Return { value } => {
                let mut instrs = Vec::new();
                if let Some(value) = value {
                    instrs.extend(self.lower_instruction(value, function_indices, current_function_index, label_stack, next_label_id)?);
                }
                instrs.push(wasm_ir::Instruction::Return);
                Ok(instrs)
            },
            Instruction::Call { function_name, args } => {
                let function_index = self.get_function_index(function_name, function_indices).unwrap();
                
                let mut instrs = Vec::new();
                for arg in args {
                    instrs.extend(self.lower_instruction(arg, function_indices, current_function_index, label_stack, next_label_id)?);
                }
                instrs.push(wasm_ir::Instruction::Call { function_index });
                Ok(instrs)
            }
            Instruction::Drop => Ok(vec![wasm_ir::Instruction::Drop]),
        }
    }

    fn lower_value(&self, value: &Value) -> wasm_ir::Instruction {
        match value {
            Value::I32(v) => wasm_ir::Instruction::I32Const { value: *v },
            Value::I64(v) => wasm_ir::Instruction::I64Const { value: *v },
            Value::F32(v) => wasm_ir::Instruction::F32Const { value: *v },
            Value::F64(v) => wasm_ir::Instruction::F64Const { value: *v },
        }
    }

    fn lower_type(&self, ty: &Type) -> wasm_ir::Type {
        match ty {
            Type::I32 => wasm_ir::Type::ValueType { value_type: wasm_ir::ValueType::NumType { num_type: wasm_ir::NumType::I32 } },
            Type::I64 => wasm_ir::Type::ValueType { value_type: wasm_ir::ValueType::NumType { num_type: wasm_ir::NumType::I64 } },
            Type::F32 => wasm_ir::Type::ValueType { value_type: wasm_ir::ValueType::NumType { num_type: wasm_ir::NumType::F32 } },
            Type::F64 => wasm_ir::Type::ValueType { value_type: wasm_ir::ValueType::NumType { num_type: wasm_ir::NumType::F64 } },
        }
    }

    fn get_local_index(&self, local_function_index: usize, name: &str) -> u32 {
        let mut index = 0;
        for (param_name, _) in &self.local_functions[local_function_index].params {
            if param_name == name {
                return index;
            }
            index += 1;
        }
        for (local_name, _) in &self.local_functions[local_function_index].locals {
            if local_name == name {
                return index;
            }
            index += 1;
        }
        
        panic!("Local not found: {}", name);
    }

    fn get_locals_names(&self, local_function_index: usize) -> Vec<String> {
        let mut names = Vec::new();
        for (param_name, _) in &self.local_functions[local_function_index].params {
            names.push(param_name.clone());
        }
        for (local_name, _) in &self.local_functions[local_function_index].locals {
            names.push(local_name.clone());
        }
        names
    }

    fn get_global_index(&self, name: &str) -> Result<u32, Vec<errors::CompileError>> {
        self.globals.iter().position(|g| g.name == name).map(|idx| idx as u32).ok_or(vec![errors::CompileError::IRLoweringError(format!("Global not found: {}", name), Position { line: 0, column: 0, index: 0 })])
    }

    fn does_it_return(&self, body: &[TypedInstruction]) -> bool {
        // Check if any instruction in the body is a return recursively
        for instr in body {
            if matches!(instr.instruction, Instruction::Return { .. }) {
                return true;
            }
            match &instr.instruction {
                Instruction::If { then_body, else_body, .. } => {
                    if self.does_it_return(then_body) && self.does_it_return(else_body) {
                        return true;
                    }
                },
                Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                    if self.does_it_return(body) {
                        return true;
                    }
                },
                _ => {}
            }
        }
        false
    }

    fn get_local_function(&self, name: &str) -> Option<&Function> {
        self.local_functions.iter().find(|f| f.name == name)   
    }

    fn get_host_import(&self, name: &str) -> Option<&HostImport> {
        self.host_imports.iter().find(|h| h.function == name)
    }

    fn get_function_local_index(&self, name: &str) -> Option<usize> {
        self.local_functions.iter().position(|f| f.name == name)
    }

    fn get_function_index(&self, name: &str, function_indices: &Vec<String>) -> Option<u32> {
        function_indices.iter().position(|n| n == name).map(|idx| idx as u32)
    }

    fn find_or_create_type_index(&self, types: &mut Vec<wasm_ir::FunctionType>, function_type: &wasm_ir::FunctionType) -> u32 {
        if let Some(idx) = types.iter().position(|t| t == function_type) {
            idx as u32
        } else {
            types.push(function_type.clone());
            (types.len() - 1) as u32
        }
    }

    #[allow(unused)]
    pub fn display(&self) {
        println!("Module:");
        for global in &self.globals {
            println!("  Global: {} (mutable: {}, type: {:?}, export: {})", global.name, global.mutable, global.global_type, global.export);
        }
        for function in &self.local_functions {
            println!("  Function: {} (export: {})", function.name, function.export);
            println!("    Params:");
            for (name, ty) in &function.params {
                println!("      {}: {:?}", name, ty);
            }
            println!("    Return type: {:?}", function.return_type);
            println!("    Locals:");
            for (name, ty) in &function.locals {
                println!("      {}: {:?}", name, ty);
            }
            println!("    Body:");
            for instr in &function.body {
                println!("      {:?} (result type: {:?})", instr.instruction, instr.result_type);
            }
        }
    }

    fn find_binop(&self, op: &BinOp, ty: &Type) -> Option<wasm_ir::Instruction> {
        match (op, ty) {
            (BinOp::Add, Type::I32) => Some(wasm_ir::Instruction::I32Add),
            (BinOp::Sub, Type::I32) => Some(wasm_ir::Instruction::I32Sub),
            (BinOp::Mul, Type::I32) => Some(wasm_ir::Instruction::I32Mul),
            (BinOp::Div, Type::I32) => Some(wasm_ir::Instruction::I32DivS),
            (BinOp::Gt, Type::I32) => Some(wasm_ir::Instruction::I32Gt),
            (BinOp::Ge, Type::I32) => Some(wasm_ir::Instruction::I32Ge),
            (BinOp::Lt, Type::I32) => Some(wasm_ir::Instruction::I32Lt),
            (BinOp::Le, Type::I32) => Some(wasm_ir::Instruction::I32Le), 
            (BinOp::Eq, Type::I32) => Some(wasm_ir::Instruction::I32Eq),
            
            (BinOp::Add, Type::I64) => Some(wasm_ir::Instruction::I64Add),
            (BinOp::Sub, Type::I64) => Some(wasm_ir::Instruction::I64Sub),
            (BinOp::Mul, Type::I64) => Some(wasm_ir::Instruction::I64Mul),
            (BinOp::Div, Type::I64) => Some(wasm_ir::Instruction::I64DivS),
            (BinOp::Gt, Type::I64) => Some(wasm_ir::Instruction::I64Gt),
            (BinOp::Ge, Type::I64) => Some(wasm_ir::Instruction::I64Ge),
            (BinOp::Lt, Type::I64) => Some(wasm_ir::Instruction::I64Lt),
            (BinOp::Le, Type::I64) => Some(wasm_ir::Instruction::I64Le), 
            (BinOp::Eq, Type::I64) => Some(wasm_ir::Instruction::I64Eq),

            (BinOp::Add, Type::F32) => Some(wasm_ir::Instruction::F32Add),
            (BinOp::Sub, Type::F32) => Some(wasm_ir::Instruction::F32Sub),
            (BinOp::Mul, Type::F32) => Some(wasm_ir::Instruction::F32Mul),
            (BinOp::Div, Type::F32) => Some(wasm_ir::Instruction::F32Div),
            (BinOp::Gt, Type::F32) => Some(wasm_ir::Instruction::F32Gt),
            (BinOp::Ge, Type::F32) => Some(wasm_ir::Instruction::F32Ge),
            (BinOp::Lt, Type::F32) => Some(wasm_ir::Instruction::F32Lt),
            (BinOp::Le, Type::F32) => Some(wasm_ir::Instruction::F32Le), 
            (BinOp::Eq, Type::F32) => Some(wasm_ir::Instruction::F32Eq),

            (BinOp::Add, Type::F64) => Some(wasm_ir::Instruction::F64Add),
            (BinOp::Sub, Type::F64) => Some(wasm_ir::Instruction::F64Sub),
            (BinOp::Mul, Type::F64) => Some(wasm_ir::Instruction::F64Mul),
            (BinOp::Div, Type::F64) => Some(wasm_ir::Instruction::F64Div),
            (BinOp::Gt, Type::F64) => Some(wasm_ir::Instruction::F64Gt),
            (BinOp::Ge, Type::F64) => Some(wasm_ir::Instruction::F64Ge),
            (BinOp::Lt, Type::F64) => Some(wasm_ir::Instruction::F64Lt),
            (BinOp::Le, Type::F64) => Some(wasm_ir::Instruction::F64Le), 
            (BinOp::Eq, Type::F64) => Some(wasm_ir::Instruction::F64Eq),
        }
    }

    fn find_unop(&self, op: &UnOp, ty: &Type) -> Option<Vec<wasm_ir::Instruction>> {
        match (op, ty) {
            (UnOp::Neg, Type::I32) => Some(vec![wasm_ir::Instruction::I32Const { value: -1 }, wasm_ir::Instruction::I32Mul]),
            (UnOp::Neg, Type::I64) => Some(vec![wasm_ir::Instruction::I64Const { value: -1 }, wasm_ir::Instruction::I64Mul]),
            (UnOp::Neg, Type::F32) => Some(vec![wasm_ir::Instruction::F32Const { value: -1.0 }, wasm_ir::Instruction::F32Mul]),
            (UnOp::Neg, Type::F64) => Some(vec![wasm_ir::Instruction::F64Const { value: -1.0 }, wasm_ir::Instruction::F64Mul]),
            (UnOp::Not, Type::I32) => Some(vec![wasm_ir::Instruction::I32Const { value: 0 }, wasm_ir::Instruction::I32Eq]),
            (UnOp::Not, Type::I64) => Some(vec![wasm_ir::Instruction::I64Const { value: 0 }, wasm_ir::Instruction::I64Eq]),
            _ => None,
        }
    }

    fn find_memory_instruction(&self, op: &Instruction, ty: &Type) -> Option<wasm_ir::Instruction> {
        match (op, ty) {
            (Instruction::Load { .. }, Type::I32) => Some(wasm_ir::Instruction::I32Load { align: 1, offset: 0 }),
            (Instruction::Store { .. }, Type::I32) => Some(wasm_ir::Instruction::I32Store { align: 1, offset: 0 }),
            
            (Instruction::Load { .. }, Type::I64) => Some(wasm_ir::Instruction::I64Load { align: 1, offset: 0 }),
            (Instruction::Store { .. }, Type::I64) => Some(wasm_ir::Instruction::I64Store { align: 1, offset: 0 }),
            
            (Instruction::Load { .. }, Type::F32) => Some(wasm_ir::Instruction::F32Load { align: 1, offset: 0 }),
            (Instruction::Store { .. }, Type::F32) => Some(wasm_ir::Instruction::F32Store { align: 1, offset: 0 }),
            
            (Instruction::Load { .. }, Type::F64) => Some(wasm_ir::Instruction::F64Load { align: 1, offset: 0 }),
            (Instruction::Store { .. }, Type::F64) => Some(wasm_ir::Instruction::F64Store { align: 1, offset: 0 }),
            
            _ => None,
        }
    }
}