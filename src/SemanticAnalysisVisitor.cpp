#include "SemanticAnalysisVisitor.h"

void SemanticAnalysisVisitor::visit(FileNode* node) {
    // Init and enter global scope
    scopeByCtx_.clear();
    current_ = nullptr;
    enterScopeFor(node);

    for (std::shared_ptr<ASTNode> statNode: node->statements) {
        statNode->accept(*this);
    }
}

void SemanticAnalysisVisitor::visit(CondNode* node) {
    // Check cond type == int
    node->ifCond->accept(*this);
    if (node->ifCond->type != ValueType::INTEGER) {
        throw std::runtime_error("TypeError: Non-numeric type '" + toString(node->ifCond->type) + "' cannot be used in conditional.");
    }

    // Eval body
    enterScopeFor(node);
    for (std::shared_ptr<ASTNode> statNode: node->body) {
        statNode->accept(*this);
    }
    exitScope();
}

void SemanticAnalysisVisitor::visit(LoopNode* node) {
    // Check cond type == int
    node->loopCond->accept(*this);
    if (node->loopCond->type != ValueType::INTEGER) {
        throw std::runtime_error("TypeError: Non-numeric type '" + toString(node->loopCond->type) + "' cannot be used in conditional.");
    }

    // Eval body
    enterScopeFor(node);
    for (std::shared_ptr<ASTNode> statNode: node->body) {
        statNode->accept(*this);
    }
    exitScope();
}

void SemanticAnalysisVisitor::visit(IntDecNode* node) {
    node->value->accept(*this);
    node->id->accept(*this);

    // Ensure type coersion is possible
    if (node->value->type != ValueType::INTEGER) {
        throw std::runtime_error("TypeError: Non-numeric type '" + toString(node->value->type) + "' cannot be declared to identifier of type int.");
    }

    current_->declare(node->id->name, ValueType::INTEGER);
}

void SemanticAnalysisVisitor::visit(VectorDecNode* node) {
    node->vectorValue->accept(*this);
    node->id->accept(*this);

    // Ensure type coersion is possible
    if (node->vectorValue->type != ValueType::VECTOR) {
        throw std::runtime_error("TypeError: Non-vec type '" + toString(node->vectorValue->type) + "' cannot be declared to identifier of type vector.");
    }

    current_->declare(node->id->name, ValueType::VECTOR);
}

void SemanticAnalysisVisitor::visit(AssignNode* node) {
    node->value->accept(*this);
    node->id->accept(*this);

    // Ensure type coersion is possible
    if (promote(node->value->type, node->id->type) != node->id->type) {
        throw std::runtime_error("TypeError: Value with type '" + toString(node->value->type) + "' cannot be assigned to identifier '" + node->id->name + "' with type '" + toString(node->id->type) + "'.");
    }    
}

void SemanticAnalysisVisitor::visit(IntNode *node) {

}

void SemanticAnalysisVisitor::visit(IdNode *node) {
    // Ensure var exists
    SymbolInfo *idInfo = current_->resolve(node->name);
    if (idInfo == nullptr) {
        throw std::runtime_error("ReferenceError: Identifier '" + node->name + "' assigned to before declaration.");
    }

    // Resolve var type
    node->type = idInfo->type;
}

void SemanticAnalysisVisitor::enterScopeFor(const ASTNode* ownerCtx) {
    // Init root
    if (current_ == nullptr) {
    root_ = std::make_unique<Scope>(nullptr);
    current_ = root_.get();
  }
  Scope* child = current_->createChild();
  scopeByCtx_[ownerCtx] = child;
  current_ = child;
}

void SemanticAnalysisVisitor::exitScope() {
  if (current_ && current_->parent()) {
    current_ = current_->parent();
  }
}