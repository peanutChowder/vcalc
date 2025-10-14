#pragma once
#include "VCalcBaseVisitor.h"  // MISSING - This defines the base class methods to override
#include "VCalcParser.h"       // MISSING - This defines the Context types
#include "antlr4-runtime.h"

using namespace vcalc;

class ASTBuilder: public VCalcBaseVisitor{
    public:
        antlrcpp::Any visitFile(VCalcParser::FileContext *ctx) override;
        antlrcpp::Any visitIntDec(VCalcParser::IntDecContext *ctx) override;
        antlrcpp::Any visitVectorDec(VCalcParser::VectorDecContext *ctx) override;
        antlrcpp::Any visitAssign(VCalcParser::AssignContext *ctx) override;
        antlrcpp::Any visitPrint(VCalcParser::PrintContext *ctx) override;
        antlrcpp::Any visitCond(VCalcParser::CondContext *ctx) override;
        antlrcpp::Any visitLoop(VCalcParser::LoopContext *ctx) override;
        
        antlrcpp::Any visitEqualityExpr(VCalcParser::EqualityExprContext *ctx) override;
        antlrcpp::Any visitComparisonExpr(VCalcParser::ComparisonExprContext *ctx) override;
        antlrcpp::Any visitAddSubExpr(VCalcParser::AddSubExprContext *ctx) override;
        antlrcpp::Any visitMulDivExpr(VCalcParser::MulDivExprContext *ctx) override;
        antlrcpp::Any visitRangeExpr(VCalcParser::RangeExprContext *ctx) override;
        antlrcpp::Any visitIndexExpr(VCalcParser::IndexExprContext *ctx) override;
        
        antlrcpp::Any visitGenerator(VCalcParser::GeneratorContext *ctx) override;
        antlrcpp::Any visitFilter(VCalcParser::FilterContext *ctx) override;
        antlrcpp::Any visitAtom(VCalcParser::AtomContext *ctx) override;
    
};