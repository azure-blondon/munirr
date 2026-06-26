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
    I32Sub,
    I32Mul,
    I32DivS,
    I32RemS,
    I32Eq,
    I32Gt,
    I32Ge,
    I32Lt,
    I32Le,
    
    I64Add,
    I64Sub,
    I64Mul,
    I64DivS,
    I64RemS,
    I64Eq,
    I64Gt,
    I64Ge,
    I64Lt,
    I64Le,
    
    F32Add,
    F32Sub,
    F32Mul,
    F32Div,
    F32Eq,
    F32Gt,
    F32Ge,
    F32Lt,
    F32Le,

    F64Add,
    F64Sub,
    F64Mul,
    F64Div,
    F64Eq,
    F64Gt,
    F64Ge,
    F64Lt,
    F64Le,
    
    I32Store { align: u32, offset: u32 },
    I32Load { align: u32, offset: u32 },
    
    I64Store { align: u32, offset: u32 },
    I64Load { align: u32, offset: u32 },

    F32Store { align: u32, offset: u32 },
    F32Load { align: u32, offset: u32 },

    F64Store { align: u32, offset: u32 },
    F64Load { align: u32, offset: u32 },

    LocalGet { id: LocalIndex },
    LocalSet { id: LocalIndex },
    GlobalGet { id: GlobalIndex },
    GlobalSet { id: GlobalIndex },
    
    Nop,
    Unreachable,
    Drop,
    Block { block_type: BlockType, body: Vec<Instruction> },
    Loop { block_type: BlockType, body: Vec<Instruction> },
    Br { label_index: LabelIndex },
    BrIf { label_index: LabelIndex },
    If { block_type: BlockType, then_body: Vec<Instruction>, else_body: Vec<Instruction> },
    Return,
    Call { function_index: FunctionIndex },
}