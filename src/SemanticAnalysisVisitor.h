#pragma once

#include "SCalcBaseVisitor.h"
#include "Scope.h"
#include "antlr4-runtime.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace scalc {

class SemanticAnalysisVisitor : public SCalcBaseVisitor {
public:
    std::any visitProg(SCalcParser::ProgContext *ctx) override;

    // statements
    std::any visitLoop(SCalcParser::LoopContext *ctx) override;
    std::any visitCondition(SCalcParser::ConditionContext *ctx) override;
    std::any visitExpr(SCalcParser::ExprContext *ctx) override;
    std::any visitAssign(SCalcParser::AssignContext *ctx) override;
    std::any visitDeclr(SCalcParser::DeclrContext *ctx) override;

    // operators
    std::any visitMulDiv(SCalcParser::MulDivContext *ctx) override;
    std::any visitAddSub(SCalcParser::AddSubContext *ctx) override;
    std::any visitCmp(SCalcParser::CmpContext *ctx) override;
    std::any visitEqNeq(SCalcParser::EqNeqContext *ctx) override;
    std::any visitAtom(SCalcParser::AtomContext *ctx) override;

    // Hand off to interpreter
    Scope* getScopeRoot() const { return root_.get(); }
    const std::unordered_map<const antlr4::ParserRuleContext*, Scope*>& scopeIndex() const { return scopeByCtx_; }

private:
    // Persistent scope tree and context index
    std::unique_ptr<Scope> root_;
    Scope* current_ = nullptr;
    std::unordered_map<const antlr4::ParserRuleContext*, Scope*> scopeByCtx_;

    void enterScopeFor(const antlr4::ParserRuleContext* ownerCtx);
    void exitScope();
};

} 
