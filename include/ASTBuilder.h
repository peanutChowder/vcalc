#pragma once
#include "VCalcBaseVisitor.h"
#include "AST.h"
#include <memory>

class ASTBuilder: public VCalcBaseVisitor{
    public:
        antlrcpp::Any visitFile(VCalcParser::FileContext *ctx) override;
        antlrcpp::Any visitStat(VCalcParser::StatContext *ctx) override;
        antlrcpp::Any visitIntDec(VCalcParser::IntDecContext *ctx) override;
        antlrcpp::Any visitVectorDec(VCalcParser::VectorDecContext *ctx) override;
        antlrcpp::Any visitAssign(VCalcParser::AssignContext *ctx) override;
        antlrcpp::Any visitPrint(VCalcParser::PrintContext *ctx) override;
        antlrcpp::Any visitCond(VCalcParser::CondContext *ctx) override;
        antlrcpp::Any visitLoop(VCalcParser::LoopContext *ctx) override;
        antlrcpp::Any visitBinaryOp(VCalcParser::BinaryOpContext *ctx) override;
        antlrcpp::Any visitGenerator(VCalcParser::GeneratorContext *ctx) override;
        antlrcpp::Any visitFilter(VCalcParser::FilterContext *ctx) override;
        antlrcpp::Any visitRange(VCalcParser::RangeContext *ctx) override;
        antlrcpp::Any visitIndex(VCalcParser::IndexContext *ctx) override;
        antlrcpp::Any visitInt(VCalcParser::IntContext *ctx) override;
        antlrcpp::Any visitId(VCalcParser::IdContext *ctx) override;
}