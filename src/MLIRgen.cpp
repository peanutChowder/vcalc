/*
Traverse AST tree and for each node emit MLIR operations
Backend sets up MLIR context, builder, and helper functions
After generating the MLIR, Backend will lower the dialects and output LLVM IR
*/

#include "AST.h"
#include "BackEnd.h"
#include "ASTVisitor.h"
#include <vector>
#include <unordered_map>
#include <string>
#include <mlir/IR/Value.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/MLIRContext.h>
#include <iostream>

class MLIRGen : public ASTVisitor {
    public:
        explicit MLIRGen(BackEnd &backend);
        // AST node visit methods
        void visit(FileNode* node) override;
        void visit(IntNode* node) override;
        void visit(IdNode* node) override;
        void visit(BinaryOpNode* node) override;
        void visit(RangeNode* node) override;
        void visit(IndexNode* node) override;
        void visit(GeneratorNode* node) override;
        void visit(FilterNode* node) override;
        void visit(IntDecNode* node) override;
        void visit(VectorDecNode* node) override;
        void visit(AssignNode* node) override;
        void visit(PrintNode* node) override;
        void visit(CondNode* node) override;
        void visit(LoopNode* node) override;

private:
    BackEnd &backend;
    mlir::OpBuilder *builder;
    mlir::ModuleOp module;
    mlir::MLIRContext *context;

    // Stack for intermediate MLIR values
    std::vector<mlir::Value> v_stack;

    // Symbol table for MLIR values
    // when variables are declared their mlir values are stored on the stack
    std::unordered_map<std::string, mlir::Value> symbols;
};

MLIRGen::MLIRGen(BackEnd &backend)
  : backend(backend),
    builder(backend.getBuilder().get()),         
    module(backend.getModule()),           
    context(&backend.getContext()) {} 

void MLIRGen::visit(FileNode* node){
    for(const std::shared_ptr<ASTNode>& stmt: node->statements){
        stmt->accept(*this);
    }
}
// each id node will have a corresponding value
void MLIRGen::visit(IdNode* node){
    // retrieve MLIR value from symbol table
    auto id = symbols.find(node->name);
    if(id != symbols.end()){
        v_stack.push_back(id->second);
    }else{
        // value doesnt exist in symbol table yet
    }
}
void MLIRGen::visit(IntNode* node){
    assert(node->type == ValueType::INTEGER);
    // create MLIR value
    mlir::Value value = builder->create<mlir::arith::ConstantOp>(backend.getLoc(), 
                            builder->getI32Type(), builder->getI32IntegerAttr(node->value)); 
    // store in stack intermediately
    v_stack.push_back(value);
}
void MLIRGen::visit(IntDecNode* node){
    // visit value of int, generate its MLIR value and push onto stack
    node->value->accept(*this); // calls visitExpr which will push value computed  value onto stack
    mlir::Value value = v_stack.back(); //receive from stack
    v_stack.pop_back(); // remove from stack
    // store in symbol table (ei x = 5)
    symbols[node->id->name] = value;
}
void MLIRGen::visit(VectorDecNode* node){
    assert(node->type == ValueType::VECTOR);
    // will visit ExprNode -> GeneratorNode/FilterNode/etc
    node->vectorValue->accept(*this);
    mlir::Value value = v_stack.back();
    v_stack.pop_back();
    symbols[node->id->name] = value;
}
void MLIRGen::visit(AssignNode* node){
    node->value->accept(*this);
    mlir::Value value = v_stack.back();
    v_stack.pop_back();
    symbols[node->id->name] = value;
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

    mlir::Value value;
    if(node->left->type == ValueType::INTEGER && node->right->type == ValueType::INTEGER){
            // left and right are Integers
        if(node->op == "+"){
            value = builder.create<mlir::arith::AddIOp>(backend.getLoc(), left, right);
        }else if(node->op == "-"){
            value = builder.create<mlir::arith::SubIOp>(backend.getLoc(), left, right);
        }else if(node->op == "*"){
            value = builder.create<mlir::arith::MulIOp>(backend.getLoc(), left, right);
        }else if(node->op == "/"){
            value = builder.create<mlir::arith::DivSIOp>(backend.getLoc(), left, right);
        }else{
            std::cout << "unsupported operator" << std::endl;
            return;
        }
        // push back result value
        v_stack.push_back(value); 
    }
    // at least on of the values is a vector
}
/*
When you visit a node that creates or manipulates a vector, emit MLIR ops to allocate memory, store values, and access elements.
*/