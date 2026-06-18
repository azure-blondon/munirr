use crate::common::position::Position;

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
    pub position: Position,
}

#[derive(Debug)]
pub struct Global {
    pub name: String,
    pub ty: Type,
    pub mutable: bool,
    pub init: TypedNode,
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

#[derive(Debug)]
#[allow(dead_code)]
pub enum TypeDef {
    Alias { name: String, ty: Type, position: Position },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    I32,
    I64,
    F32,
    F64,
    Buf(Box<Type>),
}


#[derive(Debug, Clone)]
pub enum TypedNode {
    Statement { statement: Statement },
    Expression { expression: Expression, result_type: Option<Type> },
}


#[derive(Debug, Clone)]
pub enum Statement {
    If { condition: Box<TypedNode>, then_body: Vec<TypedNode>, else_body: Vec<TypedNode>, position: Position },
    Return { value: Option<Box<TypedNode>>, position: Position },
    Expression { expression: Expression, position: Position },
    VariableDeclaration { name: String, ty: Type, init: Option<Box<TypedNode>>, position: Position },
    Block { body: Vec<TypedNode>, position: Position },
    Loop { body: Vec<TypedNode>, position: Position },
    Break { position: Position },
    Continue { position: Position },
}

#[derive(Debug, Clone)]
pub enum Expression {
    BinaryOp { op: BinOp, left: Box<TypedNode>, right: Box<TypedNode>, position: Position },
    UnaryOp { op: UnOp, operand: Box<TypedNode>, position: Position },
    Literal { value: Literal, position: Position },
    Identifier { name: String, position: Position },
    Call { function: String, args: Vec<TypedNode>, position: Position },
    BufferAccess { buffer: Box<TypedNode>, index: Box<TypedNode>, position: Position },
}

#[derive(Debug, Clone)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
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
    Character(i32),
    String(String),
}


#[derive(Debug, Clone)]
pub enum BlockType {
    Loop,
    If,
    Block,
}
