use crate::muni_ast::*;
use crate::errors::{CompileError, Position};


pub struct TypeChecker {
    
}

impl TypeChecker {
    pub fn new() -> Self {
        TypeChecker { }
    }
    
    
    pub fn check_ast(&mut self, ast: &mut Program) -> Result<(), Vec<CompileError>> {
        let mut errors = Vec::new();
            
        // check global initializations
        for global in &mut ast.module.globals {
            self.check_global_initialization(global).unwrap_or_else(|e| errors.push(e));
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
    
    
    fn check_global_initialization(&mut self, global: &mut Global) -> Result<(), CompileError> {
        self.update_expression_types(&mut global.init, Some(global.ty))?;
        if !self.is_expression_constant(&global.init) {
            return Err(CompileError::TypeCheckingError(
                format!("Global '{}' must be a constant expression", global.name),
                global.position
            ));
        }
        Ok(())
    }
    
    
    
    fn update_expression_types(&mut self, node: &mut TypedNode, wants: Option<Type>) -> Result<Type, CompileError> {
        match node {
            TypedNode::Expression { expression, result_type } => {
                let inferred_type = match expression {
                    Expression::Literal { value, position: _ } => {
                        let literal_type = match value {
                            Literal::Integer(_) => Type::I64,
                            Literal::Float(_) => Type::F64,
                        };
                        
                        if let Some(wants) = wants {
                            self.cast_type(literal_type, wants, expression.position())?
                        } else {
                            literal_type
                        }
                    }
                    Expression::UnaryOp { op: _, operand, position: _ } => {
                        let operand_type = self.update_expression_types(operand, wants.clone())?;
                        operand_type
                    }
                    Expression::BinaryOp { op, left, right, position } => {
                        let left_type = self.update_expression_types(left, wants.clone())?;
                        let right_type = self.update_expression_types(right, Some(left_type.clone()))?;
                        
                        if left_type != right_type {
                            return Err(CompileError::TypeCheckingError(
                                format!("Type mismatch in binary op: {:?} vs {:?}", left_type, right_type),
                                *position
                            ));
                        }
                        
                        match op {
                            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => left_type,
                            BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq => Type::I32,
                            BinOp::Assign => left_type,
                        }
                    }
                    Expression::Identifier { name, position } => {
                        return Err(CompileError::TypeCheckingError(
                            format!("Global initializers can only contain literals, found identifier '{}'", name),
                            *position
                        ));
                    }
                    Expression::Call { function: _, args: _, position } => {
                        return Err(CompileError::TypeCheckingError(
                            "Global initializers can only contain literals".to_string(),
                            *position
                        ));
                    }
                };
                
                *result_type = Some(inferred_type.clone());
                Ok(inferred_type)
            }
            TypedNode::Statement { statement } => Err(CompileError::TypeCheckingError("Expected an expression".to_string(), statement.position())),
        }
    }
    
    
    
    fn cast_type(&self, from: Type, to: Type, position: Position) -> Result<Type, CompileError> {
        match to {
            Type::I32 => match from {
                Type::I32 => Ok(Type::I32),
                Type::I64 => Ok(Type::I32),
                _ => Err(CompileError::TypeCheckingError(format!("Cannot cast {:?} to {:?}", from, to), position)),
            }
            Type::I64 => match from {
                Type::I32 => Ok(Type::I64),
                Type::I64 => Ok(Type::I64),
                _ => Err(CompileError::TypeCheckingError(format!("Cannot cast {:?} to {:?}", from, to), position)),
            },
            Type::F32 => match from {
                Type::F32 => Ok(Type::F32),
                Type::F64 => Ok(Type::F32),
                _ => Err(CompileError::TypeCheckingError(format!("Cannot cast {:?} to {:?}", from, to), position)),
            }
            Type::F64 => match from {
                Type::F32 => Ok(Type::F64),
                Type::F64 => Ok(Type::F64),
                _ => Err(CompileError::TypeCheckingError(format!("Cannot cast {:?} to {:?}", from, to), position)),
            }
        }
    }


    fn is_expression_constant(&self, node: &TypedNode) -> bool {
        match node {
            TypedNode::Expression { expression, result_type: _ } => {
                match expression {
                    Expression::Literal { value: _, position: _ } => true,
                    Expression::UnaryOp { op: _, operand, position: _ } => self.is_expression_constant(operand),
                    Expression::BinaryOp { op: _, left, right, position: _ } => {
                        self.is_expression_constant(left) && self.is_expression_constant(right)
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

}