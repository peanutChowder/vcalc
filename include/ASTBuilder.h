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
        std::any visitFile(VCalcParser::FileContext *ctx) override;
        std::any visitIntDec(VCalcParser::IntDecContext *ctx) override;
        std::any visitVectorDec(VCalcParser::VectorDecContext *ctx) override;
        std::any visitAssign(VCalcParser::AssignContext *ctx) override;
        std::any visitPrint(VCalcParser::PrintContext *ctx) override;
        std::any visitCond(VCalcParser::CondContext *ctx) override;
        std::any visitLoop(VCalcParser::LoopContext *ctx) override;
        
        std::any visitEqualityExpr(VCalcParser::EqualityExprContext *ctx) override;
        std::any visitComparisonExpr(VCalcParser::ComparisonExprContext *ctx) override;
        std::any visitAddSubExpr(VCalcParser::AddSubExprContext *ctx) override;
        std::any visitMulDivExpr(VCalcParser::MulDivExprContext *ctx) override;
        std::any visitRangeExpr(VCalcParser::RangeExprContext *ctx) override;
        std::any visitIndexExpr(VCalcParser::IndexExprContext *ctx) override;
        
        std::any visitGenerator(VCalcParser::GeneratorContext *ctx) override;
        std::any visitFilter(VCalcParser::FilterContext *ctx) override;
        std::any visitAtom(VCalcParser::AtomContext *ctx) override;
    
};