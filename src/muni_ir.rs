use std::{collections::HashMap};

use crate::wasm_ir::{self, ExportDescriptor};

#[derive(Debug)]
pub struct Module {
    pub functions: Vec<Function>,
    pub globals: Vec<Global>,
}


#[derive(Debug)]
pub struct Function {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub return_type: Option<Type>,
    pub body: Vec<TypedInstruction>,
    pub locals: Vec<(String, Type)>,
    pub export: bool,
}

#[derive(Debug)]
pub struct Global {
    pub name: String,
    pub mutable: bool,
    pub global_type: Type,
    pub init: Vec<TypedInstruction>,
    pub export: bool,
}


#[derive(Debug, Clone)]
pub enum Type {
    I32,
    I64,
    F32,
    F64,
}

#[derive(Debug, Clone)]
pub struct TypedInstruction {
    pub instruction: Instruction,
    pub result_type: Option<Type>,
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Const { value: Value },
    BinaryOp { op: BinOp, left: Box<TypedInstruction>, right: Box<TypedInstruction> },
    VarGet { name: String },
    VarSet { name: String, value: Box<TypedInstruction> },
    If { condition: Box<TypedInstruction>, then_body: Vec<TypedInstruction>, else_body: Vec<TypedInstruction> },
    Loop { label: String, body: Vec<TypedInstruction> },
    Block { label: String, body: Vec<TypedInstruction> },
    Break { value: u32 },
    Return { value: Option<Box<TypedInstruction>> },
    Call { function_name: String, args: Vec<TypedInstruction> },
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
pub enum Value {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}






impl Module {
    pub fn lower(&self) -> wasm_ir::Module {
        let mut module = wasm_ir::Module {
            types: Vec::new(),
            functions: Vec::new(),
            globals: Vec::new(),
            exports: Vec::new(),
        };
        
        let function_indices: HashMap<String, usize> = self.functions.iter().enumerate().map(|(idx, func)| (func.name.clone(), idx)).collect();

        for global in &self.globals {
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
                init: global.init.iter().flat_map(|instr| self.lower_instruction(instr, &function_indices, None, &mut Vec::new(), &mut 0)).collect(),
            });
            if global.export {
                module.exports.push(wasm_ir::Export {
                    name: global.name.clone(),
                    descriptor: ExportDescriptor::GlobalIndex(module.globals.len() as u32 - 1),
                });
            }
        }


        for (idx, function) in self.functions.iter().enumerate() {
            // Build type for this function
            let outputs = match &function.return_type {
                None => vec![],
                Some(ty) => vec![self.lower_type(ty)],
            };
            module.types.push(wasm_ir::FunctionType {
                inputs: function.params.iter().map(|(_, ty)| self.lower_type(ty)).collect(),
                outputs,
            });
            
            // Build function with the same index
            module.functions.push(wasm_ir::Function {
                function_type: idx as u32,  // Type index always equals function index
                locals: function.locals.iter().map(|(_, ty)| self.lower_type(ty)).collect(),
                body: function.body.iter().flat_map(|instr| self.lower_instruction(instr, &function_indices, Some(idx), &mut Vec::new(), &mut 0)).collect(),
            });
            
            if function.export {
                module.exports.push(wasm_ir::Export {
                    name: function.name.clone(),
                    descriptor: ExportDescriptor::FunctionIndex(idx as u32),
                });
            }
        }

        
        module
    }


    fn lower_instruction(
        &self,
        instruction: &TypedInstruction,
        function_indices: &HashMap<String, usize>,
        current_function_index: Option<usize>,
        label_stack: &mut Vec<(String, u32)>,
        next_label_id: &mut u32,
    ) -> Vec<wasm_ir::Instruction> {
        match &instruction.instruction {
            Instruction::Const { value } => vec![self.lower_value(&value)],
            Instruction::BinaryOp { op, left, right } => {
                let mut instrs = self.lower_instruction(left, function_indices, current_function_index, label_stack, next_label_id);
                instrs.extend(self.lower_instruction(right, function_indices, current_function_index, label_stack, next_label_id));
                instrs.push(match op {
                    BinOp::Add => wasm_ir::Instruction::I32Add,
                    BinOp::Sub => wasm_ir::Instruction::I32Sub,
                    BinOp::Mul => wasm_ir::Instruction::I32Mul,
                    BinOp::Div => wasm_ir::Instruction::I32DivS,
                    BinOp::Gt => wasm_ir::Instruction::I32Gt,
                    BinOp::Ge => wasm_ir::Instruction::I32Ge,
                    BinOp::Lt => wasm_ir::Instruction::I32Lt,
                    BinOp::Le => wasm_ir::Instruction::I32Le, 
                    BinOp::Eq => wasm_ir::Instruction::I32Eq, // TODO handle other types
                });
                instrs
            },
            Instruction::VarGet { name } => {
                if let Some(func_idx) = current_function_index {
                    if self.functions[func_idx].params.iter().any(|(param_name, _)| param_name == name) || self.functions[func_idx].locals.iter().any(|(local_name, _)| local_name == name) {
                        return vec![wasm_ir::Instruction::LocalGet { id: self.get_local_index(current_function_index, name) }];
                    }
                }

                vec![wasm_ir::Instruction::GlobalGet { id: self.get_global_index(name) }]
            },
            Instruction::VarSet { name, value } => {
                if let Some(func_idx) = current_function_index {
                    if self.functions[func_idx].params.iter().any(|(param_name, _)| param_name == name) || self.functions[func_idx].locals.iter().any(|(local_name, _)| local_name == name) {
                        let mut instrs = self.lower_instruction(value, function_indices, current_function_index, label_stack, next_label_id);
                        instrs.push(wasm_ir::Instruction::LocalSet { id: self.get_local_index(current_function_index, name) });
                        return instrs;
                    }
                }
                vec![wasm_ir::Instruction::GlobalSet { id: self.get_global_index(name) }]
            },
            Instruction::If { condition, then_body, else_body } => {
                let mut instrs = self.lower_instruction(condition, function_indices, current_function_index, label_stack, next_label_id);
                
                instrs.push(wasm_ir::Instruction::If {
                    block_type: wasm_ir::BlockType::ValueTypes { value_types: Vec::new() },
                    then_body: then_body.iter().flat_map(|instr| self.lower_instruction(instr, function_indices, current_function_index, label_stack, next_label_id)).collect(),
                    else_body: else_body.iter().flat_map(|instr| self.lower_instruction(instr, function_indices, current_function_index, label_stack, next_label_id)).collect(),
                });
                
                let both_branches_return = self.does_it_return(then_body) && self.does_it_return(else_body);

                if both_branches_return {
                    instrs.push(wasm_ir::Instruction::Unreachable);
                }

                instrs
            },
            Instruction::Block { label, body } => {
                let mut instrs = Vec::new();
                let label_id = *next_label_id;
                *next_label_id += 1;
                label_stack.push((label.clone(), label_id));
                let lowered = body.iter().flat_map(|instr| self.lower_instruction(instr, function_indices, current_function_index, label_stack, next_label_id)).collect();
                label_stack.pop();
                instrs.push(wasm_ir::Instruction::Block {
                    block_type: wasm_ir::BlockType::ValueTypes { value_types: Vec::new() },
                    body: lowered,
                });
                if self.does_it_return(body) {
                    instrs.push(wasm_ir::Instruction::Unreachable);
                }
                instrs
            },
            Instruction::Loop { label, body } => {
                let mut instrs = Vec::new();
                let label_id = *next_label_id;
                *next_label_id += 1;
                label_stack.push((label.clone(), label_id));
                let lowered = body.iter().flat_map(|instr| self.lower_instruction(instr, function_indices, current_function_index, label_stack, next_label_id)).collect();
                label_stack.pop();
                instrs.push(wasm_ir::Instruction::Loop {
                    block_type: wasm_ir::BlockType::ValueTypes { value_types: Vec::new() },
                    body: lowered,
                });
                if self.does_it_return(body) {
                    instrs.push(wasm_ir::Instruction::Unreachable);
                }
                instrs
            },
            Instruction::Break { value } => {
                let mut instrs = Vec::new();
                instrs.push(wasm_ir::Instruction::Br { label_index: *value });
                instrs
            },
            Instruction::Return { value } => {
                let mut instrs = Vec::new();
                if let Some(value) = value {
                    instrs.extend(self.lower_instruction(value, function_indices, current_function_index, label_stack, next_label_id));
                }
                instrs.push(wasm_ir::Instruction::Return);
                instrs
            },
            Instruction::Call { function_name, args } => {
                let function_index = *function_indices.get(function_name).expect("Function not found"); 
                let mut instrs = Vec::new();
                for arg in args {
                    instrs.extend(self.lower_instruction(arg, function_indices, current_function_index, label_stack, next_label_id));
                }
                instrs.push(wasm_ir::Instruction::Call { function_index: function_index as u32 });
                instrs
            }
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

    fn lower_type(&self, ty: &Type) -> wasm_ir::ValueType {
        match ty {
            Type::I32 => wasm_ir::ValueType::NumType { num_type: wasm_ir::NumType::I32 },
            Type::I64 => wasm_ir::ValueType::NumType { num_type: wasm_ir::NumType::I64 },
            Type::F32 => wasm_ir::ValueType::NumType { num_type: wasm_ir::NumType::F32 },
            Type::F64 => wasm_ir::ValueType::NumType { num_type: wasm_ir::NumType::F64 },
        }
    }

    fn get_local_index(&self, current_function_index: Option<usize>, name: &str) -> u32 {
        let mut index = 0;
        if let Some(func_idx) = current_function_index {
            for (param_name, _) in &self.functions[func_idx].params {
                if param_name == name {
                    return index;
                }
                index += 1;
            }
            for (local_name, _) in &self.functions[func_idx].locals {
                if local_name == name {
                    return index;
                }
                index += 1;
            }
        }
        panic!("Local not found: {}", name);
    }

    fn get_locals_names(&self, current_function_index: Option<usize>) -> Vec<String> {
        let mut names = Vec::new();
        if let Some(func_idx) = current_function_index {
            for (param_name, _) in &self.functions[func_idx].params {
                names.push(param_name.clone());
            }
            for (local_name, _) in &self.functions[func_idx].locals {
                names.push(local_name.clone());
            }
        }
        names
    }

    fn get_global_index(&self, name: &str) -> u32 {
        self.globals.iter().position(|g| g.name == name).expect("Global not found") as u32
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

    #[allow(unused)]
    pub fn display(&self) {
        println!("Module:");
        for global in &self.globals {
            println!("  Global: {} (mutable: {}, type: {:?}, export: {})", global.name, global.mutable, global.global_type, global.export);
        }
        for function in &self.functions {
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
}