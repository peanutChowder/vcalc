/*
Traverse AST tree and for each node emit MLIR operations
Backend sets up MLIR context, builder, and helper functions
After generating the MLIR, Backend will lower the dialects and output LLVM IR
*/
#include "MLIRgen.h";

MLIRgen::MLIRgen(Backend &backend): backend(backend), 
        builder(*backend.builder), module(backend.module),
        context(backend.context){}

// used to store intermediate values
std::vector<mlir::Value> v_stack;
// symbol table is used to store MLIR values for variables

void MLIRGen::visit(FileNode* node){
    for(auto &stmt:node->statements){
        stmt->accept(*this); //from AST
    }
}

void MLIRGen::visit(IntNode* node){
    auto value = builder.create<mlir::arth::ConstantOp>(backend.loc, builder.getI32Type(), builder.getI32IntegerAttr(node->value))
    // store value in symbol table?
    // store value in stack
    v_stack.push_back(value);
}

void MLIRGen::visit(IdNode* node){
    // retrieve MLIR value from symbol table
    auto id = symbolTable.find(node->id);
    if(id != symbolTable.end()){
        mlir::Value value = id->second;
        v_stack.push_back(it->second);
    }else{
        // value doesnt exist in symbol table yet
    }
}
//! this must also support vector operations
// AddlOp -> add
void MLIRGen::visit(BinaryOpNode* node){
    // recursively visit left and right children nodes of BinaryOpNode
    node->left->accept(*this); // pushed 1st
    node->right->accept(*this); // pushed 2nd
    
    auto right = v_stack.back(); 
    v_stack.pop_back();
    auto left = v_stack.back();
    v_stack.pop_back();

    // type check to see if left and/or right are vectors
    // a runtime method could handle element-wise operations
    mlir::Value value;
    // left and right are Integers
    if(node->op == "+"){
        value = builder.create<mlir::arth::AddIOp>(backend.loc, left, right);
    }else if(node->op == "-"){
        value = builder.create<mlir::arth::SubIOp>(backend.loc, left, right);
    }else if(node->op == "*"){
        value = builder.create<mlir::arth::MulIOp>(backend.loc, left, right);
    }else if(node->op == "/"){
        value = builder.create<mlir::arth::DivIOp>(backend.loc, left, right);
    }else{
        std::out << "unsupported operator"
    }

    v_stack.push_back(value);   
}
/*
When you visit a node that creates or manipulates a vector, emit MLIR ops to allocate memory, store values, and access elements.
*/