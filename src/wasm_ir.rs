pub type TypeIndex = u32;
pub type FunctionIndex = u32;
pub type LocalIndex = u32;
pub type GlobalIndex = u32;
pub type LabelIndex = u32;
pub type MemoryIndex = u32;

#[derive(Debug, Clone)]
pub struct Module {
    pub types: Vec<FunctionType>,
    pub globals: Vec<Global>,
    pub memories: Vec<Memory>,
    pub functions: Vec<Function>,
    pub host_imports: Vec<HostImport>,
    pub exports: Vec<Export>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub type_index: TypeIndex,
    pub locals: Vec<ValueType>,
    pub body: Expression,
}

#[derive(Debug, Clone)]
pub struct Global {
    pub mutable: Mutability,
    pub global_type: ValueType,
    pub init: Expression,
}
#[derive(Debug, Clone)]
pub enum Mutability {
    Mutable,
    Immutable,
}

#[derive(Debug, Clone)]
pub struct Memory {
    pub min_pages: u32,
    pub max_pages: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct HostImport {
    pub module: String,
    pub function: String,
    pub type_index: TypeIndex,
}

#[derive(Debug, Clone)]
pub struct Export {
    pub name: String,
    pub descriptor: ExportDescriptor,
}

#[derive(Debug, Clone)]
pub enum ExportDescriptor {
    FunctionIndex(FunctionIndex),
    GlobalIndex(GlobalIndex),
    MemoryIndex(MemoryIndex),
}

#[derive(Debug, Clone)]
pub enum Type {
    ValueType { value_type: ValueType },
}

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub enum ValueType {
    NumType { num_type: NumType },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionType {
    pub inputs: ResultType,
    pub outputs: ResultType,
}
pub type ResultType = Vec<ValueType>;

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub enum NumType {
    I32,
    I64,
    F32,
    F64,
}

pub type Expression = Vec<Instruction>;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum BlockType {
    TypeIndex { index: TypeIndex },
    ValueTypes { value_types: Vec<ValueType> },
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Instruction {
    I32Const { value: i32 },
    I64Const { value: i64 },
    F32Const { value: f32 },
    F64Const { value: f64 },
    I32Add,
    I64Add,
    F32Add,
    F64Add,
    I32Gt,
    I32Ge,
    I32Lt,
    I32Le,
    I32Sub,
    I32Mul,
    I32DivS,
    I32Eq,
    
    LocalGet { id: LocalIndex },
    LocalSet { id: LocalIndex },
    GlobalGet { id: GlobalIndex },
    GlobalSet { id: GlobalIndex },
    
    Nop,
    Unreachable,
    Block { block_type: BlockType, body: Vec<Instruction> },
    Loop { block_type: BlockType, body: Vec<Instruction> },
    Br { label_index: LabelIndex },
    BrIf { label_index: LabelIndex },
    If { block_type: BlockType, then_body: Vec<Instruction>, else_body: Vec<Instruction> },
    Return,
    Call { function_index: FunctionIndex },
}













pub trait Emittable {
    fn emit(&mut self, out: &mut Vec<u8>);
}


fn encode_u32(mut n: u32, out: &mut Vec<u8>) {
    loop {
        let mut byte = (n & 0x7F) as u8;
        n >>= 7;
        if n != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if n == 0 {
            break;
        }
    }
}
fn encode_i32(n: i32, out: &mut Vec<u8>) {
    let mut more = true;
    let mut value = n as u32;
    let size = 32;

    while more {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;

        let sign_bit = (byte & 0x40) != 0;

        if (value == 0 && !sign_bit) || (value == (!0 >> (size - 7)) && sign_bit) {
            more = false;
        } else {
            byte |= 0x80;
        }

        out.push(byte);
    }
}
fn encode_i64(n: i64, out: &mut Vec<u8>) {
    let mut more = true;
    let mut value = n as u64;
    let size = 64;

    while more {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;

        let sign_bit = (byte & 0x40) != 0;

        if (value == 0 && !sign_bit) || (value == (!0 >> (size - 7)) && sign_bit) {
            more = false;
        } else {
            byte |= 0x80;
        }

        out.push(byte);
    }
}


fn write_section(section_id: u8, payload: &[u8], out: &mut Vec<u8>) {
    out.push(section_id);
    encode_u32(payload.len() as u32, out);
    out.extend_from_slice(payload);
}



impl Emittable for Module {
    fn emit(&mut self, out: &mut Vec<u8>) {
        // WASM Magic Number and Version
        out.extend_from_slice(&[0x00, 0x61, 0x73, 0x6d]); // "\0asm"
        out.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]); // version 1


        // Type Section
        let mut type_section = Vec::new();
        encode_u32(self.types.len() as u32, &mut type_section);
        for func_type in &mut self.types {
            func_type.emit(&mut type_section);
        }
        write_section(1, &type_section, out);


        // Host Import Section
        let mut import_section = Vec::new();
        encode_u32(self.host_imports.len() as u32, &mut import_section);
        for import in self.host_imports.clone() {
            // Module name
            encode_u32(import.module.len() as u32, &mut import_section);
            import_section.extend_from_slice(import.module.as_bytes());
            
            // Function name
            encode_u32(import.function.len() as u32, &mut import_section);
            import_section.extend_from_slice(import.function.as_bytes());
            
            // Import kind: 0x00 = function
            import_section.push(0x00);
            
            encode_u32(import.type_index, &mut import_section);
        }
        write_section(2, &import_section, out);

        
        // Function Section
        let mut function_section = Vec::new();
        encode_u32(self.functions.len() as u32, &mut function_section);
        for func in &self.functions {
            encode_u32(func.type_index, &mut function_section);
        }
        write_section(3, &function_section, out);
        
        
        // Memory Section
        let mut memory_section = Vec::new();
        encode_u32(self.memories.len() as u32, &mut memory_section);

        for memory in &self.memories {
            match memory.max_pages {
                Some(max) => {
                    encode_u32(0x01, &mut memory_section); // flags (has max)
                    encode_u32(memory.min_pages, &mut memory_section);
                    encode_u32(max, &mut memory_section);
                }
                None => {
                    encode_u32(0x00, &mut memory_section); // flags (no max)
                    encode_u32(memory.min_pages, &mut memory_section);
                }
            }
        }

        write_section(5, &memory_section, out);

        
        // Global Section
        let mut global_section = Vec::new();
        encode_u32(self.globals.len() as u32, &mut global_section);
        for global in &mut self.globals {
            global.emit(&mut global_section);
        }
        write_section(6, &global_section, out);


        // Export Section
        let mut export_section = Vec::new();
        encode_u32(self.exports.len() as u32, &mut export_section);
        for export in &mut self.exports {
            encode_u32(export.name.len() as u32, &mut export_section);
            export_section.extend_from_slice(export.name.as_bytes());
            match &export.descriptor {
                ExportDescriptor::FunctionIndex(index) => {
                    export_section.push(0x00); // Function export
                    // Adjust index: imports come first in the function index space
                    let adjusted_index = self.host_imports.len() as u32 + index;
                    encode_u32(adjusted_index, &mut export_section);
                }
                ExportDescriptor::MemoryIndex(index) => {
                    export_section.push(0x02); // Memory export
                    encode_u32(*index, &mut export_section);
                }
                ExportDescriptor::GlobalIndex(index) => {
                    export_section.push(0x03); // Global export
                    encode_u32(*index, &mut export_section);
                }
            }
        }
        write_section(7, &export_section, out);

        // Code Section
        let mut code_section = Vec::new();
        encode_u32(self.functions.len() as u32, &mut code_section);
        for func in &mut self.functions {
            func.emit(&mut code_section);
        }
        write_section(10, &code_section, out);

    }

}


impl Emittable for FunctionType {
    fn emit(&mut self, out: &mut Vec<u8>) {
        out.push(0x60); // Function type form
        // Emit inputs
        encode_u32(self.inputs.len() as u32, out);
        for input in &mut self.inputs {
            input.emit(out);
        }
        // Emit outputs
        encode_u32(self.outputs.len() as u32, out);
        for output in &mut self.outputs {
            output.emit(out);
        }
    }
}

impl Emittable for ValueType {
    fn emit(&mut self, out: &mut Vec<u8>) {
        match self {
            ValueType::NumType { num_type } => {
                match num_type {
                    NumType::I32 => out.push(0x7F),
                    NumType::I64 => out.push(0x7E),
                    NumType::F32 => out.push(0x7D),
                    NumType::F64 => out.push(0x7C),
                }
            }
        }
    }
}

impl Emittable for Function {
    fn emit(&mut self, out: &mut Vec<u8>) {
        let mut func_body = Vec::new();
        // Emit locals
        let mut local_counts = std::collections::HashMap::new();
        for local in &self.locals {
            *local_counts.entry(local).or_insert(0) += 1;
        }
        encode_u32(local_counts.len() as u32, &mut func_body);
        for (local_type, count) in local_counts {
            encode_u32(count, &mut func_body);
            local_type.clone().emit(&mut func_body);
        }
        // Emit body
        for instr in &mut self.body {
            instr.emit(&mut func_body);
        }
        func_body.push(0x0B); // End opcode

        // Emit function size
        encode_u32(func_body.len() as u32, out);
        out.extend_from_slice(&func_body);
    }
}

impl Emittable for Instruction {
    fn emit(&mut self, out: &mut Vec<u8>) {
        match self {
            Instruction::I32Const { value } => {
                out.push(0x41);
                encode_i32(*value, out);
            }
            Instruction::I64Const { value } => {
                out.push(0x42);
                encode_i64(*value, out);
            }
            Instruction::F32Const { value } => {
                out.push(0x43);
                out.extend_from_slice(&value.to_le_bytes());
            }
            Instruction::F64Const { value } => {
                out.push(0x44);
                out.extend_from_slice(&value.to_le_bytes());
            }
            Instruction::I32Add => out.push(0x6A),
            Instruction::I64Add => out.push(0x7C),
            Instruction::F32Add => out.push(0x92),
            Instruction::F64Add => out.push(0xA0),
            Instruction::I32Gt => out.push(0x4A),
            Instruction::I32Ge => out.push(0x4E),
            Instruction::I32Lt => out.push(0x4C),
            Instruction::I32Le => out.push(0x4D),
            Instruction::I32Sub => out.push(0x6B),
            Instruction::I32Mul => out.push(0x6C),
            Instruction::I32DivS => out.push(0x6D),
            Instruction::I32Eq => out.push(0x46),

            Instruction::LocalGet { id } => {
                out.push(0x20);
                encode_u32(*id, out);
            }
            Instruction::LocalSet { id } => {
                out.push(0x21);
                encode_u32(*id, out);
            }
            Instruction::GlobalGet { id } => {
                out.push(0x23);
                encode_u32(*id, out);
            }
            Instruction::GlobalSet { id } => {
                out.push(0x24);
                encode_u32(*id, out);
            }
            Instruction::Nop => out.push(0x01),
            Instruction::Unreachable => out.push(0x00),
            Instruction::Block { block_type, body } => {
                out.push(0x02);
                match block_type {
                    BlockType::TypeIndex { index } => {
                        encode_u32(*index, out);
                    }
                    BlockType::ValueTypes { value_types } => {
                        if value_types.is_empty() {
                            out.push(0x40);
                        } else {
                            encode_u32(value_types.len() as u32, out);
                            for vt in value_types {
                                vt.emit(out);
                            }
                        }
                    }
                }
                for instr in body {
                    instr.emit(out);
                }
                out.push(0x0B); // End opcode
            }
            Instruction::Loop { block_type, body } => {
                out.push(0x03);
                match block_type {
                    BlockType::TypeIndex { index } => {
                        encode_u32(*index, out);
                    }
                    BlockType::ValueTypes { value_types } => {
                        if value_types.is_empty() {
                            out.push(0x40);
                        } else {
                            encode_u32(value_types.len() as u32, out);
                            for vt in value_types {
                                vt.emit(out);
                            }
                        }
                        
                    }
                }
                for instr in body {
                    instr.emit(out);
                }
                out.push(0x0B); // End opcode
            }
            Instruction::Br { label_index } => {
                out.push(0x0C);
                encode_u32(*label_index, out);
            }
            Instruction::BrIf { label_index } => {
                out.push(0x0D);
                encode_u32(*label_index, out);
            }
            Instruction::If { block_type, then_body, else_body } => {
                out.push(0x04);
                match block_type {
                    BlockType::TypeIndex { index } => {
                        encode_u32(*index, out);
                    }
                    BlockType::ValueTypes { value_types } => {
                        if value_types.is_empty() {
                            out.push(0x40);
                        } else {
                            encode_u32(value_types.len() as u32, out);
                            for vt in value_types {
                                vt.emit(out);
                            }
                        }
                    }
                }
                for instr in then_body {
                    instr.emit(out);
                }
                if !else_body.is_empty() {
                    out.push(0x05); // Else opcode
                    for instr in else_body {
                        instr.emit(out);
                    }
                }
                out.push(0x0B); // End opcode
            }
            Instruction::Return => out.push(0x0F),
            Instruction::Call { function_index } => {
                out.push(0x10);
                encode_u32(*function_index, out);
            }
        }
    }
}


impl Emittable for Global {
    fn emit(&mut self, out: &mut Vec<u8>) {
        self.global_type.emit(out);
        match self.mutable {
            Mutability::Mutable => out.push(0x01),
            Mutability::Immutable => out.push(0x00),
        }
        for instr in &mut self.init {
            instr.emit(out);
        }
        out.push(0x0B); // End opcode
    }
}

impl Module {




    #[allow(unused)]
    pub fn display(&self) -> String {
        let mut s = String::new();
        s.push_str("Module {\n");
        s.push_str("  Types:\n");
        for (i, t) in self.types.iter().enumerate() {
            s.push_str(&format!("    {}: {:?}\n", i, t));
        }
        s.push_str("  Functions:\n");
        for (i, f) in self.functions.iter().enumerate() {
            s.push_str(&format!("    {}: {:?}\n", i, f));
        }
        s.push_str("  Globals:\n");
        for (i, g) in self.globals.iter().enumerate() {
            s.push_str(&format!("    {}: {:?}\n", i, g));
        }
        s.push_str("  Exports:\n");
        for export in &self.exports {
            s.push_str(&format!("    {:?}\n", export));
        }
        s.push_str("}");
        s
    }
}