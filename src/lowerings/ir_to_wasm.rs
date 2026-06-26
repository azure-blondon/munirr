use std::collections::HashMap;
use std::thread::current;
use crate::irs::new_ir;
use crate::irs::wasm_ir as wasm;

pub struct LoweringContext {
    pub local_map: HashMap<String, u32>,
    pub next_local_index: u32,
    pub struct_layouts: HashMap<String, Vec<(String, u64, new_ir::Type)>>,
    pub function_types: HashMap<(Vec<new_ir::Type>, Option<new_ir::Type>), wasm::TypeIndex>,
    pub output: wasm::Module,
}

impl LoweringContext {
    pub fn new() -> Self {
        Self {
            local_map: HashMap::new(),
            next_local_index: 0,
            struct_layouts: HashMap::new(),
            function_types: HashMap::new(),
            output: wasm::Module {
                types: vec![],
                globals: vec![],
                memories: vec![],
                functions: vec![],
                host_imports: vec![],
                exports: vec![],
            },
        }
    }
    fn size_of_type(&self, ty: &new_ir::Type) -> u64 {
        match ty {
            new_ir::Type::I32 | new_ir::Type::F32 => 4,
            new_ir::Type::I64 | new_ir::Type::F64 => 8,
            new_ir::Type::Buf(_) => 8, // (ptr + length)
            new_ir::Type::Struct(s) => {
                if let Some(layout) = self.struct_layouts.get(&s.name) {
                    if let Some((_, last_offset, last_ty)) = layout.last() {
                        return last_offset + self.size_of_type(last_ty);
                    }
                    return 0;
                }
                panic!("Struct layout not found for {}", s.name);
            },
            new_ir::Type::Enum(_) => 8, // i64
        }
    }

    fn compute_layouts(&mut self, types: &[new_ir::TypeDeclaration]) {
        for declaration in types {
            if let new_ir::TypeDeclaration::Struct(s) = declaration {
                let mut offset: u64 = 0;
                let mut layout: Vec<(String, u64, new_ir::Type)> = Vec::new();
                for (field_name, field_type) in &s.fields {
                    let size = self.size_of_type(field_type); // TODO: implement
                    layout.push((field_name.clone(), offset, field_type.clone()));
                    offset += size;
                }
                self.struct_layouts.insert(s.name.clone(), layout);
            }
        }
    }

    fn get_or_create_function_type_index(&mut self, params: &[new_ir::Type], return_type: Option<&new_ir::Type>) -> wasm::TypeIndex {
        let signature = (params.to_vec(), return_type.cloned());
        
        if let Some(idx) = self.function_types.get(&signature) {
            return *idx;
        }

        let mut inputs = Vec::new();
        for p in params {
            inputs.push(self.convert_type(p));
        }
        
        let mut outputs = Vec::new();
        if let Some(rt) = return_type {
            outputs.push(self.convert_type(rt));
        }

        let new_idx = self.output.types.len() as u32;
        self.output.types.push(wasm::FunctionType {
            inputs,
            outputs,
        });

        self.function_types.insert(signature, new_idx);
        return new_idx;
    }

    fn convert_type(&mut self, ty: &new_ir::Type) -> wasm::ValueType {
        match ty {
            new_ir::Type::I32 => wasm::ValueType::NumType { num_type: wasm::NumType::I32 },
            new_ir::Type::I64 => wasm::ValueType::NumType { num_type: wasm::NumType::I64 },
            new_ir::Type::F32 => wasm::ValueType::NumType { num_type: wasm::NumType::F32 },
            new_ir::Type::F64 => wasm::ValueType::NumType { num_type: wasm::NumType::F64 },
            // Buffers/Structs/Enums lower to I32 (or I64) in wasm
            new_ir::Type::Buf(_) | new_ir::Type::Struct(_) | new_ir::Type::Enum(_) => {
                wasm::ValueType::NumType { num_type: wasm::NumType::I32 }
            }
        }
    }

    pub fn lower_module(&mut self, module: &new_ir::Module) {
        // 1. Compute Struct Layouts
        self.compute_layouts(&module.type_declarations);

        // 2. Register Types & Lower Imports
        for import in &module.host_imports {
            let type_idx = self.get_or_create_function_type_index(&import.params, import.return_type.as_ref());
            self.output.host_imports.push(wasm::HostImport {
                module: import.module.clone(),
                function: import.function.clone(),
                type_index: type_idx,
            });
        }

        // 3. Lower Globals
        for global in &module.globals {
            let w_type = self.convert_type(&global.global_type);
            // Lower init expression (must be constant in Wasm MVP)
            let init_expr = self.lower_constant_expression(&global.init); 
            
            self.output.globals.push(wasm::Global {
                mutable: if global.mutable { wasm::Mutability::Mutable } else { wasm::Mutability::Immutable },
                global_type: w_type,
                init: init_expr,
            });
        }

        // 4. Lower Memories
        for mem in &module.memories {
            self.output.memories.push(wasm::Memory {
                min_pages: mem.min_size, // Convert bytes to pages, min 1
                max_pages: mem.max_size,
            });
        }

        // 5. Lower Functions
        // We need to know function indices. 
        // Imports come first, then defined functions.
        let import_count = self.output.host_imports.len() as u32;

        for (func_idx, func) in module.functions.iter().enumerate() {
            let param_types: Vec<new_ir::Type> = func.params.iter().map(|(_, t)| t.clone()).collect();
            let type_idx = self.get_or_create_function_type_index(&param_types, func.return_type.as_ref());
            
            self.lower_function(func, type_idx, import_count + func_idx as u32);

        }

        // 6. Handle Exports
        let mut current_func_idx = import_count;
        for func in &module.functions {
            if func.export {
                self.output.exports.push(wasm::Export {
                    name: func.name.clone(),
                    descriptor: wasm::ExportDescriptor::FunctionIndex(current_func_idx),
                });
            }
            current_func_idx += 1;
        }
    }

    fn lower_constant_expression(&mut self, init: &[new_ir::TypedInstruction]) -> wasm::Expression {
        // We assume init is a single Const instruction
        if init.len() != 1 {
            panic!("Multiple instructions in constant is not supported")
        }
        let mut expr = Vec::new();
        for instr in init {
            if let new_ir::Instruction::Const { value } = &instr.instruction {
                match value {
                    new_ir::Value::I32(v) => expr.push(wasm::Instruction::I32Const { value: *v }),
                    new_ir::Value::I64(v) => expr.push(wasm::Instruction::I64Const { value: *v }),
                    new_ir::Value::F32(v) => expr.push(wasm::Instruction::F32Const { value: *v }),
                    new_ir::Value::F64(v) => expr.push(wasm::Instruction::F64Const { value: *v }),
                }
            }
        }
        expr
    }

    fn lower_function(&mut self, func: &new_ir::Function, type_idx: wasm::TypeIndex, _func_index: u32) {
        // Reset state for new function
        self.local_map.clear();
        self.next_local_index = 0;

        // Map params to locals 0..N
        for (i, (name, _)) in func.params.iter().enumerate() {
            self.local_map.insert(name.clone(), i as u32);
        }
        // Map locals to locals N..M
        let mut wasm_locals = Vec::new();
        for (name, ty) in func.locals.iter() {
            self.local_map.insert(name.clone(), self.next_local_index);
            wasm_locals.push(self.convert_type(ty));
            self.next_local_index += 1;
        }

        // TODO: reorder locals for efficiency? (later)

        let mut current_body: Vec<wasm::Instruction> = Vec::new();

        for instruction in &func.body {
            current_body.extend(self.lower_instruction(instruction));
        }
        self.output.functions.push(wasm::Function {
            type_index: type_idx,
            locals: wasm_locals,
            body: current_body,

        })
    }

    fn lower_instruction(&mut self, instruction: &new_ir::TypedInstruction) -> Vec<wasm::Instruction> {
        let mut result = vec![];
        match &instruction.instruction {
            new_ir::Instruction::Const { value } => {
                match value {
                    new_ir::Value::I32(v) => result.push(wasm::Instruction::I32Const { value: *v }),
                    new_ir::Value::I64(v) => result.push(wasm::Instruction::I64Const { value: *v }),
                    new_ir::Value::F32(v) => result.push(wasm::Instruction::F32Const { value: *v }),
                    new_ir::Value::F64(v) => result.push(wasm::Instruction::F64Const { value: *v }),
                }
            },
            new_ir::Instruction::UnaryOp { op, operand } => {
                let mut instrs = self.lower_instruction(operand);
                let operand_type = operand.result_type.as_ref();
                if operand_type.is_none() {
                    panic!("Unary operation operand must have a type");
                }
                let operand_type = operand_type.unwrap();
                let unop_instrs = find_unop(op, operand_type).unwrap();
                instrs.extend(unop_instrs);
            }
            _ => panic!("not implemented yet"),
        }
        result
    }
}




fn find_unop(op: &new_ir::UnOp, ty: &new_ir::Type) -> Option<Vec<wasm::Instruction>> {
    match (op, ty) {
        (new_ir::UnOp::Negation, new_ir::Type::I32) => Some(vec![wasm::Instruction::I32Const { value: -1 }, wasm::Instruction::I32Mul]),
        (new_ir::UnOp::Negation, new_ir::Type::I64) => Some(vec![wasm::Instruction::I64Const { value: -1 }, wasm::Instruction::I64Mul]),
        (new_ir::UnOp::Negation, new_ir::Type::F32) => Some(vec![wasm::Instruction::F32Const { value: -1.0 }, wasm::Instruction::F32Mul]),
        (new_ir::UnOp::Negation, new_ir::Type::F64) => Some(vec![wasm::Instruction::F64Const { value: -1.0 }, wasm::Instruction::F64Mul]),
        (new_ir::UnOp::BitwiseNot, new_ir::Type::I32) => Some(vec![wasm::Instruction::I32Const { value: 0 }, wasm::Instruction::I32Eq]),
        (new_ir::UnOp::BitwiseNot, new_ir::Type::I64) => Some(vec![wasm::Instruction::I64Const { value: 0 }, wasm::Instruction::I64Eq]),
        _ => None,
    }
}