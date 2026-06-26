use crate::common::position::Position;


#[derive(Debug)]
pub struct Module {
    pub functions: Vec<Function>,
    pub globals: Vec<Global>,
    pub host_imports: Vec<HostImport>,
    pub memories: Vec<Memory>,
    pub type_declarations: Vec<TypeDeclaration>,
}

#[derive(Debug)]
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
pub struct Global {
    pub name: String,
    pub mutable: bool,
    pub global_type: Type,
    pub init: Vec<TypedInstruction>,
    pub export: bool,
    pub position: Position,
}


#[derive(Debug)]
pub struct HostImport {
    pub module: String,
    pub function: String,
    pub params: Vec<Type>,
    pub return_type: Option<Type>,
    pub position: Position,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    I32,
    I64,
    F32,
    F64,
    Buf(Box<Type>),
    Struct(Struct),
    Enum(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Memory {
    pub name: String,
    pub min_size: u32,
    pub max_size: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeDeclaration {
    Struct(Struct),
    Enum(Enum),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Struct {
    pub name: String,
    pub fields: Vec<(String, Type)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Enum {
    pub name: String,
    pub variants: Vec<(String, i64)>,
}

#[derive(Debug, Clone)]
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
    // High level block of code
    Block { body: Vec<TypedInstruction>, result_type: Option<Type> },
    If { condition: Box<TypedInstruction>, then_body: Vec<TypedInstruction>, else_body: Vec<TypedInstruction> },
    // High level infinite loop
    Loop { body: Vec<TypedInstruction> },
    Break { condition: Option<Box<TypedInstruction>> },
    Continue { condition: Option<Box<TypedInstruction>> },
    Return { value: Option<Box<TypedInstruction>> },
    Call { function_name: String, args: Vec<TypedInstruction> },
    Load { address: Box<TypedInstruction>, memory: String },
    Store { address: Box<TypedInstruction>, value: Box<TypedInstruction>, memory: String },
    Alloc { block_size: u32, amount: Box<TypedInstruction>, memory: String },
    FieldLoad { name: String, field_name: String, value: Box<TypedInstruction> },
    FieldStore { name: String, field_name: String },
    Cast { from: Type, to: Type, value: Box<TypedInstruction> },
    Drop,
    Unreachable,
}

#[derive(Debug, Clone)]
pub enum BinOp {
    Addition,
    Subtraction,
    Multiplication,
    Division,
    Modulo,
    GreaterThan,
    GreaterEqual,
    LesserThan,
    LesserEqual,
    Equal,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    ShiftRight,
    ShiftLeft,
}

#[derive(Debug, Clone)]
pub enum UnOp {
    Negation,
    BitwiseNot,
}

#[derive(Debug, Clone)]
pub enum Value {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}