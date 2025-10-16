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

    // Ensure type coersion is possible
    if (node->value->type != ValueType::INTEGER) {
        throw std::runtime_error("TypeError: Non-numeric type '" + toString(node->value->type) + "' cannot be declared to identifier of type int.");
    }

    current_->declare(node->id->name, ValueType::INTEGER);
    node->id->accept(*this);
}

void SemanticAnalysisVisitor::visit(VectorDecNode* node) {
    node->vectorValue->accept(*this);
    current_->declare(node->id->name, ValueType::VECTOR);
    node->id->accept(*this);

    // Ensure type coersion is possible
    if (node->vectorValue->type != ValueType::VECTOR) {
        throw std::runtime_error("TypeError: Non-vec type '" + toString(node->vectorValue->type) + "' cannot be declared to identifier of type vector.");
    }
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
    node->type = ValueType::INTEGER;
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

void SemanticAnalysisVisitor::visit(BinaryOpNode *node) {
    node->left->accept(*this);
    node->right->accept(*this);

    if (node->left->type == ValueType::UNKNOWN) {
        throw std::runtime_error("TypeError: Invalid left operand type '" + toString(node->left->type) + "' for operator '" + node->op + "'");
    }

    if (node->right->type == ValueType::UNKNOWN) {
        throw std::runtime_error("TypeError: Invalid right operand type '" + toString(node->right->type) + "' for operator '" + node->op + "'");
    }

    // If vectors are involved then promote to vector
    node->type = ValueType::INTEGER;
    if (node->left->type == ValueType::VECTOR || node->right->type == ValueType::VECTOR) {
        node->type = ValueType::VECTOR;
    }
}

void SemanticAnalysisVisitor::visit(RangeNode *node) {
    node->start->accept(*this);
    node->end->accept(*this);

    if (node->start->type != ValueType::INTEGER) {
        throw std::runtime_error("TypeError: Invalid left operand type '" + toString(node->start->type) + "' for range operator.");
    }

    if (node->end->type != ValueType::INTEGER) {
        throw std::runtime_error("TypeError: Invalid right operand type '" + toString(node->end->type) + "' for range operator.");
    }

    node->type = ValueType::VECTOR;
}

void SemanticAnalysisVisitor::visit(IndexNode* node) {
    node->array->accept(*this);
    node->index->accept(*this);

    if (node->array->type != ValueType::VECTOR) {
        throw std::runtime_error("TypeError: Non-vector type '" + toString(node->array->type) + "' used with index operator.");
    }

    if (node->index->type != ValueType::INTEGER) {
        throw std::runtime_error("TypeError: Non-int index type '" + toString(node->index->type) + "' used as index.");
    }

    node->type = ValueType::INTEGER;
}

void SemanticAnalysisVisitor::visit(GeneratorNode *node) {
    // "mini" scope to handle declr of domain variable
    enterScopeFor(node);
    current_->declare(node->id->name, ValueType::INTEGER);
    node->id->accept(*this);
    node->domain->accept(*this);
    node->body->accept(*this);

    if (node->domain->type != ValueType::VECTOR) {
        throw std::runtime_error("TypeError: Non-vec type '" + toString(node->domain->type) + "' used as domain in generator.");
    }

    if (node->body->type != ValueType::INTEGER) {
        throw std::runtime_error("TypeError: Non-int type '" + toString(node->body->type) + "' used as expression in generator.");
    }
    exitScope();

    node->type = ValueType::VECTOR;
}

void SemanticAnalysisVisitor::visit(FilterNode *node) {
    // "mini" scope to handle declr of domain variable
    enterScopeFor(node);
    current_->declare(node->id->name, ValueType::INTEGER);
    node->id->accept(*this);
    node->domain->accept(*this);
    node->predicate->accept(*this);

    if (node->domain->type != ValueType::VECTOR) {
        throw std::runtime_error("TypeError: Non-vec type '" + toString(node->domain->type) + "' used as domain in filter.");
    }

    if (node->predicate->type != ValueType::INTEGER) {
        throw std::runtime_error("TypeError: Non-int type '" + toString(node->predicate->type) + "' used as predicate in filter.");
    }
    exitScope();

    node->type = ValueType::VECTOR;
}

void SemanticAnalysisVisitor::visit(PrintNode *node) {
    node->printExpr->accept(*this);

    if (node->printExpr->type == ValueType::UNKNOWN) {
        throw std::runtime_error("TypeError: Invalid operand type '" + toString(node->printExpr->type) + "' for print.");
    }
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