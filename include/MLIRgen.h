#pragma once

#include "AST.h"
#include "ASTVisitor.h"
#include "BackEnd.h"

#include <string>
#include <unordered_map>
#include <vector>

#include <mlir/IR/Value.h>

// MLIRGen traverses the AST and emits MLIR operations
class MLIRGen : public ASTVisitor {
public:
    explicit MLIRGen(BackEnd& backend);

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
    mlir::Value popValue();
    void pushValue(mlir::Value value);

    [[maybe_unused]] BackEnd& backend_;
    mlir::OpBuilder& builder_;
    mlir::ModuleOp module_;
    mlir::MLIRContext& context_;
    mlir::Location loc_;

    // Stack for intermediate MLIR values
    std::vector<mlir::Value> v_stack_;

    // Symbol table for variable values
    std::unordered_map<std::string, mlir::Value> symbolTable_;
};
