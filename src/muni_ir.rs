use crate::wasm_ir::{self, ExportDescriptor};
use crate::errors::Position;

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
    pub fn lower(&mut self) -> wasm_ir::Module {
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
                    None => panic!("Function not found: {}", function),
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

            module.functions.push(wasm_ir::Function {
                type_index,
                locals: function.locals.iter().map(|(_, ty)| match self.lower_type(ty) {
                    wasm_ir::Type::ValueType { value_type } => value_type,
                }).collect(),
                body: function.body.iter().flat_map(|instr| self.lower_instruction(instr, &function_indices, Some(idx), &mut Vec::new(), &mut 0)).collect(),
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
        
        module
    }


    fn lower_instruction(
        &self,
        instruction: &TypedInstruction,
        function_indices: &Vec<String>,
        current_function_index: Option<usize>,
        label_stack: &mut Vec<(String, u32)>,
        next_label_id: &mut u32,
    ) -> Vec<wasm_ir::Instruction> {
        match &instruction.instruction {
            Instruction::Const { value } => vec![self.lower_value(&value)],
            Instruction::UnaryOp { op, operand } => {
                let mut instrs = self.lower_instruction(operand, function_indices, current_function_index, label_stack, next_label_id);
                match op {
                    UnOp::Neg => {
                        instrs.push(wasm_ir::Instruction::I32Const { value: -1 });
                        instrs.push(wasm_ir::Instruction::I32Mul);
                    }
                    UnOp::Not => {
                        instrs.push(wasm_ir::Instruction::I32Const { value: 0 });
                        instrs.push(wasm_ir::Instruction::I32Eq);
                    }
                }
                instrs
                
            }
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
            Instruction::Load { address } => {
                let mut instrs = self.lower_instruction(address, function_indices, current_function_index, label_stack, next_label_id);
                instrs.push(wasm_ir::Instruction::I32Load { align: 1, offset: 0 });
                instrs
            },
            Instruction::Store { address, value } => {
                let mut instrs = self.lower_instruction(address, function_indices, current_function_index, label_stack, next_label_id);
                instrs.extend(self.lower_instruction(value, function_indices, current_function_index, label_stack, next_label_id));
                instrs.push(wasm_ir::Instruction::I32Store { align: 1, offset: 0 });
                instrs
            },

            Instruction::Alloc { size } => {
                let mut instrs = Vec::new();
                
                // Evaluate size once and store it
                let size_instrs = self.lower_instruction(size, function_indices, current_function_index, label_stack, next_label_id);
                
                // ptr = load _heap_ptr
                instrs.push(wasm_ir::Instruction::GlobalGet { id: self.get_global_index("_heap_ptr") });
                instrs.push(wasm_ir::Instruction::GlobalSet { id: self.get_global_index("_temp_ptr") });
                
                // Store length at ptr: *ptr = size
                instrs.push(wasm_ir::Instruction::GlobalGet { id: self.get_global_index("_temp_ptr") });
                instrs.extend(size_instrs.clone());  // Use the cached size
                instrs.push(wasm_ir::Instruction::I32Store { align: 1, offset: 0 });
                
                // Increment _heap_ptr by (4 + size)
                instrs.push(wasm_ir::Instruction::GlobalGet { id: self.get_global_index("_heap_ptr") });
                instrs.push(wasm_ir::Instruction::I32Const { value: 4 });
                instrs.extend(size_instrs.clone());  // Use the cached size again
                instrs.push(wasm_ir::Instruction::I32Add);
                instrs.push(wasm_ir::Instruction::I32Add);
                instrs.push(wasm_ir::Instruction::GlobalSet { id: self.get_global_index("_heap_ptr") });
                
                // Return ptr + 4 (data start, skipping length)
                instrs.push(wasm_ir::Instruction::GlobalGet { id: self.get_global_index("_temp_ptr") });
                instrs.push(wasm_ir::Instruction::I32Const { value: 4 });
                instrs.push(wasm_ir::Instruction::I32Add);
                
                instrs
            }

            Instruction::VarGet { name } => {
                if let Some(func_idx) = current_function_index {
                    let local_func_idx = func_idx - self.host_imports.len();
                    if self.get_locals_names(local_func_idx).iter().any(|local_name| local_name == name) {
                        return vec![wasm_ir::Instruction::LocalGet { id: self.get_local_index(local_func_idx, name) }];
                    }
                }

                vec![wasm_ir::Instruction::GlobalGet { id: self.get_global_index(name) }]
            },
            Instruction::VarSet { name, value } => {
                if let Some(func_idx) = current_function_index {
                    let local_func_idx = func_idx - self.host_imports.len();
                    if self.local_functions[local_func_idx].params.iter().any(|(param_name, _)| param_name == name) || self.local_functions[local_func_idx].locals.iter().any(|(local_name, _)| local_name == name) {
                        let mut instrs = self.lower_instruction(value, function_indices, current_function_index, label_stack, next_label_id);
                        instrs.push(wasm_ir::Instruction::LocalSet { id: self.get_local_index(local_func_idx, name) });
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
                let function_index = self.get_function_index(function_name, function_indices).unwrap();
                
                let mut instrs = Vec::new();
                for arg in args {
                    instrs.extend(self.lower_instruction(arg, function_indices, current_function_index, label_stack, next_label_id));
                }
                instrs.push(wasm_ir::Instruction::Call { function_index });
                instrs
            }
            Instruction::Drop => vec![wasm_ir::Instruction::Drop],
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
}