#include "SemanticAnalysisVisitor.h"

void SemanticAnalysisVisitor::visit(FileNode* node) {
    scopeByCtx_.clear();
    root_ = std::make_unique<Scope>(nullptr);
    current_ = root_.get();
    scopeByCtx_[node] = current_;
    
    for (auto statNode: node->statements) {
        statNode->accept(*this);
    }
}