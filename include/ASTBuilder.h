#pragma once
#include "VCalcBaseVisitor.h" 
#include "VCalcParser.h"       
#include "antlr4-runtime.h"

using namespace vcalc;
/*
Class converts Parse tree produced by ANTLR into AST Tree
*/
class ASTBuilder: public VCalcBaseVisitor{
    public:
        std::unique_ptr<ASTNode> visitFile(VCalcParser::FileContext *ctx) override;
        std::unique_ptr<IntDecNode> visitIntDec(VCalcParser::IntDecContext *ctx) override;
        std::unique_ptr<VectorDecNode> visitVectorDec(VCalcParser::VectorDecContext *ctx) override;
        std::unique_ptr<AssignNode> visitAssign(VCalcParser::AssignContext *ctx) override;
        std::unique_ptr<PrintNode> visitPrint(VCalcParser::PrintContext *ctx) override;
        std::unique_ptr<CondNode> visitCond(VCalcParser::CondContext *ctx) override;
        std::unique_ptr<LoopNode> visitLoop(VCalcParser::LoopContext *ctx) override;
        
        std::unique_ptr<BinaryOpNode> visitEqualityExpr(VCalcParser::EqualityExprContext *ctx) override;
        std::unique_ptr<BinaryOpNode> visitComparisonExpr(VCalcParser::ComparisonExprContext *ctx) override;
        std::unique_ptr<BinaryOpNode> visitAddSubExpr(VCalcParser::AddSubExprContext *ctx) override;
        std::unique_ptr<BinaryOpNode> visitMulDivExpr(VCalcParser::MulDivExprContext *ctx) override;
        std::unique_ptr<RangeNode> visitRangeExpr(VCalcParser::RangeExprContext *ctx) override;
        std::unique_ptr<IndexNode> visitIndexExpr(VCalcParser::IndexExprContext *ctx) override;
        
        std::unique_ptr<GeneratorNode> visitGenerator(VCalcParser::GeneratorContext *ctx) override;
        std::unique_ptr<FilterNode> visitFilter(VCalcParser::FilterContext *ctx) override;
        std::unique_ptr<ExprNode> visitAtom(VCalcParser::AtomContext *ctx) override;
    
};