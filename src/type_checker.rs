use std::collections::HashMap;
use crate::muni_ast::*;
use crate::errors::{CompileError, Position};


pub struct TypeChecker {
    current_function: Option<String>,
    scopes: Vec<Scope>,
    functions: Vec<(String, FunctionSignature)>,
}

#[derive(Debug, Clone)]
struct FunctionSignature {
    pub params: Vec<(String, Type)>,
    pub return_type: Option<Type>,
}

struct Scope {
    pub variables: HashMap<String, Type>,
}


impl TypeChecker {
    pub fn new() -> Self {
        TypeChecker { 
            current_function: None,
            scopes: Vec::new(),
            functions: Vec::new(),
        }
    }
    
    fn push_scope(&mut self) {
        self.scopes.push(Scope { variables: HashMap::new() });
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }


    fn define_variable(&mut self, name: String, ty: Type, position: Position) -> Result<(), CompileError> {
        let current_scope = self.scopes.last_mut().unwrap();
        if current_scope.variables.contains_key(&name) {
            return Err(CompileError::TypeCheckingError(
                format!("Variable '{}' already defined in this scope", name),
                position
            ));
        }
        current_scope.variables.insert(name, ty);
        Ok(())
    }
    
    fn lookup_function(&self, name: &str) -> Option<&FunctionSignature> {
        self.functions.iter().find(|(func_name, _)| func_name == name).map(|(_, sig)| sig)
    }
    
    pub fn check_ast(&mut self, ast: &mut Program) -> Result<(), Vec<CompileError>> {
        let mut errors = Vec::new();

        self.collect_function_signatures(ast);

        
        // global scope
        self.push_scope();

        // check global initializations
        for global in &mut ast.module.globals {
            self.check_global_initialization(global).unwrap_or_else(|e| errors.push(e));
            self.define_variable(global.name.clone(), global.ty, global.position).unwrap_or_else(|e| errors.push(e));
        }

        // check function bodies
        for function in &mut ast.module.functions {
            self.current_function = Some(function.name.clone());
            self.push_scope();
            for (param_name, param_type) in &function.params {
                self.define_variable(param_name.clone(), *param_type, function.position).unwrap_or_else(|e| errors.push(e));
            }
            for instr in &mut function.body {
                self.check_node(instr).unwrap_or_else(|e| errors.push(e));
            }
            self.pop_scope();
        }
        
        self.pop_scope();

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn collect_function_signatures(&mut self, ast: &Program) {

        for import in &ast.module.host_imports {
            self.functions.push((import.function.clone(), FunctionSignature {
                params: import.params.iter().enumerate().map(|(i, ty)| (format!("arg{}", i), *ty)).collect(),
                return_type: import.return_type,
            }));
        }

        for function in &ast.module.functions {
            self.functions.push((function.name.clone(), FunctionSignature {
                params: function.params.clone(),
                return_type: function.return_type,
            }));
        }
    }
    
    
    fn check_global_initialization(&mut self, global: &mut Global) -> Result<(), CompileError> {
        self.update_expression_types(&mut global.init, Some(global.ty), true)?;
        if !self.is_expression_constant(&global.init) {
            return Err(CompileError::TypeCheckingError(
                format!("Global '{}' must be a constant expression", global.name),
                global.position
            ));
        }
        Ok(())
    }

    fn check_node(&mut self, node: &mut TypedNode) -> Result<(), CompileError> {
        match node {
            TypedNode::Statement { statement } => {
                self.check_statement(statement)
            }
            TypedNode::Expression { expression: _, result_type: _ } => {
                self.update_expression_types(node, None, false).map(|_| ())
            }
        }
    }

    fn check_statement(&mut self, statement: &mut Statement) -> Result<(), CompileError> {
        match statement {
            Statement::If { condition, then_body, else_body, position: _ } => {
            self.update_expression_types(condition, Some(Type::I32), false)?;
                    for stmt in then_body {
                        self.check_node(stmt)?;
                    }
                    for stmt in else_body {
                        self.check_node(stmt)?;
                    }
                }
                Statement::Loop { body, position: _ } => {
                    for stmt in body {
                        self.check_node(stmt)?;
                    }
                }
                Statement::Block { body, position: _ } => {
                    self.push_scope();
                    for stmt in body {
                        self.check_node(stmt)?;
                    }
                    self.pop_scope();
                }
                Statement::Return { value, position: _ } => {
                    if let Some(expr) = value {
                        if self.current_function.is_none() {
                            return Err(CompileError::TypeCheckingError(
                                "Return statement outside of function".to_string(),
                                statement.position()
                            ));
                        }
                        

                        let current_func_sig = self.lookup_function(self.current_function.as_ref().unwrap());

                        if current_func_sig.is_none() {
                            return Err(CompileError::TypeCheckingError(
                                format!("Current function '{}' not found in signature list", self.current_function.as_ref().unwrap()),
                                statement.position()
                            ));
                        }
                        let current_func_sig = current_func_sig.unwrap();

                        self.update_expression_types(expr, current_func_sig.return_type, false)?;
                    }
                }
                Statement::Break { position: _ } => {}
                Statement::Continue { position: _ } => {}
                Statement::VariableDeclaration { name, ty, init, position } => {
                    if let Some(init_expr) = init {
                        self.update_expression_types(init_expr, Some(*ty), false)?;
                    }
                    self.define_variable(name.clone(), *ty, *position)?;
                }
                Statement::Expression { expression, position: _ } => {
                    let mut checked_expr = TypedNode::Expression { expression: expression.clone(), result_type: None };

                    self.update_expression_types(&mut checked_expr, None, false)?;
                    
                    if let TypedNode::Expression { expression: checked_expression, result_type: _ } = checked_expr {
                        *expression = checked_expression;
                    }
                }
            }
        Ok(())
    }

    
    
    
    fn update_expression_types(&mut self, node: &mut TypedNode, wants: Option<Type>, const_expr: bool) -> Result<Option<Type>, CompileError> {
        let result_type: &mut Option<Type>;
        
        let expression = match node {
            TypedNode::Expression { expression, result_type: res } => {
                result_type = res;
                expression
            }
            TypedNode::Statement { statement } => {
                return Err(CompileError::TypeCheckingError(
                    "Expected expression but found statement".to_string(),
                    statement.position()
                ));
            }
            
        };

        let inferred_type = match expression {
            Expression::Literal { value, position: _ } => {
                let literal_type = match value {
                    Literal::Integer(_) => Type::I64,
                    Literal::Float(_) => Type::F64,
                    Literal::Character(_) => Type::I32,
                };
                
                if let Some(wants) = wants {
                    Some(self.cast_type(literal_type, wants, expression.position())?)
                } else {
                    Some(literal_type)
                }
            }
            Expression::UnaryOp { op: _, operand, position: _ } => {
                let operand_type = self.update_expression_types(operand, wants.clone(), const_expr)?;
                operand_type
            }
            Expression::BinaryOp { op, left, right, position } => {
                let left_type = self.update_expression_types(left, wants.clone(), const_expr)?;
                let left_type = if let Some(wants) = wants {
                    self.cast_type(left_type.ok_or_else(|| CompileError::TypeCheckingError(
                        "Could not infer type of left operand".to_string(),
                        *position
                    ))?, wants, *position)?
                } else {
                    left_type.ok_or_else(|| CompileError::TypeCheckingError(
                        "Could not infer type of left operand".to_string(),
                        *position
                    ))?
                };
                let right_type = self.update_expression_types(right, Some(left_type.clone()), const_expr)?;
                
                let right_type = if let Some(wants) = wants {
                    self.cast_type(right_type.ok_or_else(|| CompileError::TypeCheckingError(
                        "Could not infer type of right operand".to_string(),
                        *position
                    ))?, wants, *position)?
                } else {
                    right_type.ok_or_else(|| CompileError::TypeCheckingError(
                        "Could not infer type of right operand".to_string(),
                        *position
                    ))?
                };

                if left_type != right_type {
                    return Err(CompileError::TypeCheckingError(
                        format!("Type mismatch in binary op: {:?} vs {:?}", left_type, right_type),
                        *position
                    ));
                }
                
                
                let result = match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => left_type,
                    BinOp::Gt | BinOp::Lt | BinOp::Ge | BinOp::Le | BinOp::Eq => Type::I32,
                    BinOp::Assign => left_type,
                };
                Some(result)
            }
            Expression::Identifier { name, position } => {
                if const_expr {
                    return Err(CompileError::TypeCheckingError(
                        format!("Global initializers can only contain literals, found identifier '{}'", name),
                        *position
                    ));
                } else {
                    let mut found_type = None;
                    for scope in self.scopes.iter().rev() {
                        if let Some(ty) = scope.variables.get(name) {
                            found_type = Some(ty.clone());
                            break;
                        }
                    }
                    if let Some(ty) = found_type {
                        Some(ty)
                    } else {
                        return Err(CompileError::TypeCheckingError(
                            format!("Undefined variable '{}'", name),
                            *position
                        ));
                    }
                }
            }
            Expression::Call { function: function_name, args: arguments, position } => {
                if const_expr {
                    return Err(CompileError::TypeCheckingError(
                    "Global initializers can only contain literals".to_string(),
                    *position));
                } 

                let func_sig_data = self.lookup_function(function_name).cloned();
                if let Some(func_sig) = func_sig_data {
                    // Check argument count
                    if arguments.len() != func_sig.params.len() {
                        return Err(CompileError::TypeCheckingError(
                            format!("Function '{}' expects {} arguments but got {}", function_name, func_sig.params.len(), arguments.len()),
                            *position
                        ));
                    }

                    // Check argument types
                    for (arg_expr, (_, param_type)) in arguments.iter_mut().zip(func_sig.params.iter()) {
                        self.update_expression_types(arg_expr, Some(*param_type), const_expr)?;
                    }
                    
                    //println!("Function '{}', with type {:?}", function_name, func_sig.return_type);

                    func_sig.return_type
                
                
                } else {
                    return Err(CompileError::TypeCheckingError(
                        format!("Undefined function '{}'", function_name),
                        *position
                    ));
                }
                
            }
        };

        *result_type = inferred_type.clone();
        
        Ok(inferred_type)
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