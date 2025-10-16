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