/*
Traverse AST tree and for each node emit MLIR operations
Backend sets up MLIR context, builder, and helper functions
After generating the MLIR, Backend will lower the dialects and output LLVM IR
*/
#include "MLIRgen.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include <stdexcept>

MLIRGen::MLIRGen(BackEnd& backend)
    : backend_(backend),
      builder_(*backend.getBuilder()),
      module_(backend.getModule()),
      context_(backend.getContext()),
      loc_(backend.getLoc()) {}

mlir::Value MLIRGen::popValue() {
    if (v_stack.empty()) {
        throw std::runtime_error("MLIRGen internal error: value stack underflow.");
    }
    mlir::Value value = v_stack.back();
    v_stack.pop_back();
    return value;
}

void MLIRGen::pushValue(mlir::Value value) {
    if (!value) {
        throw std::runtime_error("MLIRGen internal error: attempting to push empty value onto stack.");
    }
    v_stack.push_back(value);
}

void MLIRGen::visit(FileNode* node) {
    for (const auto& stmt : node->statements) {
        if (stmt) {
            stmt->accept(*this);
        }
    }
}

void MLIRGen::visit(IntNode* node) {
    auto type = builder_.getI32Type();
    auto attr = builder_.getI32IntegerAttr(node->value);
    auto constant = builder_.create<mlir::arith::ConstantOp>(loc_, type, attr);
    pushValue(constant.getResult());
}

void MLIRGen::visit(IdNode* node) {
    auto it = symbolTable_.find(node->name);
    if (it == symbolTable_.end()) {
        throw std::runtime_error("MLIRGen error: identifier '" + node->name + "' used before assignment.");
    }
    pushValue(it->second);
}

void MLIRGen::visit(BinaryOpNode* node) {
    node->left->accept(*this);
    node->right->accept(*this);

    mlir::Value right = popValue();
    mlir::Value left = popValue();

    mlir::Value result;
    if (node->op == "+") {
        result = builder_.create<mlir::arith::AddIOp>(loc_, left, right);
    } else if (node->op == "-") {
        result = builder_.create<mlir::arith::SubIOp>(loc_, left, right);
    } else if (node->op == "*") {
        result = builder_.create<mlir::arith::MulIOp>(loc_, left, right);
    } else if (node->op == "/") {
        result = builder_.create<mlir::arith::DivSIOp>(loc_, left, right);
    } else {
        throw std::runtime_error("MLIRGen error: unsupported binary operator '" + node->op + "'.");
    }

    pushValue(result);
}

void MLIRGen::visit(RangeNode* node) {
    node->start->accept(*this);
    node->end->accept(*this);
    popValue();
    popValue();
    // Range support is not implemented yet; keep placeholder.
}

void MLIRGen::visit(IndexNode* node) {
    node->array->accept(*this);
    node->index->accept(*this);
    popValue();
    popValue();
    // Indexing support is not implemented yet.
}

void MLIRGen::visit(GeneratorNode* node) {
    node->domain->accept(*this);
    popValue();
    // Generator support is not implemented yet.
}

void MLIRGen::visit(FilterNode* node) {
    node->domain->accept(*this);
    node->predicate->accept(*this);
    popValue();
    popValue();
    // Filter support is not implemented yet.
}

void MLIRGen::visit(IntDecNode* node) {
    node->value->accept(*this);
    mlir::Value value = popValue();
    symbolTable_[node->id->name] = value;
}

void MLIRGen::visit(VectorDecNode* node) {
    node->vectorValue->accept(*this);
    mlir::Value value = popValue();
    symbolTable_[node->id->name] = value;
}

void MLIRGen::visit(AssignNode* node) {
    node->value->accept(*this);
    mlir::Value value = popValue();
    symbolTable_[node->id->name] = value;
}

void MLIRGen::visit(PrintNode* node) {
    node->printExpr->accept(*this);
    popValue();
    // Printing support would lower to a printf call; not implemented yet.
}

void MLIRGen::visit(CondNode* node) {
    node->ifCond->accept(*this);
    popValue();
    for (const auto& stmt : node->body) {
        if (stmt) {
            stmt->accept(*this);
        }
    }
}

void MLIRGen::visit(LoopNode* node) {
    node->loopCond->accept(*this);
    popValue();
    for (const auto& stmt : node->body) {
        if (stmt) {
            stmt->accept(*this);
        }
    }
}
