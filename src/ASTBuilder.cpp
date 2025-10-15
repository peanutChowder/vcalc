#include "ASTBuilder.h"
#include "AST.h"
#include "antlr4-runtime.h"
#include <memory>

using namespace vcalc;
using namespace antlr4;

std::unique_ptr<ASTNode> ASTBuilder::visitFile(VCalcParser::FileContext *ctx) {
    std::vector<std::unique_ptr<ASTNode>> statements;
    for (auto statCtx : ctx->stat()) {
        auto stmt = visit(statCtx);
        if (stmt) {
            statements.push_back(std::move(stmt));
        }
    }
    return std::make_unique<FileNode>(std::move(statements));
}

std::unique_ptr<AssignNode> ASTBuilder::visitAssign(VCalcParser::AssignContext *ctx) {
    std::string id = ctx->ID()->getText();
    auto expr = visit(ctx->expr());
    return std::make_unique<AssignNode>(id, std::move(expr));
}

std::unique_ptr<IntDecNode> ASTBuilder::visitIntDec(VCalcParser::IntDecContext *ctx) {
    std::string id = ctx->ID()->getText();
    auto expr = visit(ctx->expr());
    return std::make_unique<IntDecNode>(id, std::move(expr));
}

std::unique_ptr<VectorDecNode> ASTBuilder::visitVectorDec(VCalcParser::VectorDecContext *ctx) {
    std::string id = ctx->ID()->getText();
    auto expr = visit(ctx->expr());
    return std::make_unique<VectorDecNode>(id, std::move(expr));
}

std::unique_ptr<PrintNode> ASTBuilder::visitPrint(VCalcParser::PrintContext *ctx) {
    auto expr = visit(ctx->expr());
    return std::make_unique<PrintNode>(std::move(expr));
}

std::unique_ptr<CondNode> ASTBuilder::visitCond(VCalcParser::CondContext *ctx) {
    auto condExpr = visit(ctx->expr());
    std::vector<std::unique_ptr<ASTNode>> body;
    for (auto statCtx : ctx->stat()) {
        auto stmt = visit(statCtx);
        if (stmt) {
            body.push_back(std::move(stmt));
        }
    }
    return std::make_unique<CondNode>(std::move(condExpr), std::move(body));
}

std::unique_ptr<LoopNode> ASTBuilder::visitLoop(VCalcParser::LoopContext *ctx) {
    auto condExpr = visit(ctx->expr());
    std::vector<std::unique_ptr<ASTNode>> body;
    for (auto statCtx : ctx->stat()) {
        auto stmt = visit(statCtx);
        if (stmt) {
            body.push_back(std::move(stmt));
        }
    }
    return std::make_unique<LoopNode>(std::move(condExpr), std::move(body));
}

std::unique_ptr<BinaryOpNode> ASTBuilder::visitEqualityExpr(VCalcParser::EqualityExprContext *ctx) {
    if (ctx->comparisonExpr().size() == 1) {
        return visitComparisonExpr(ctx->comparisonExpr(0));
    }
    auto left = visitComparisonExpr(ctx->comparisonExpr(0));
    auto right = visitComparisonExpr(ctx->comparisonExpr(1));
    std::string op = ctx->op->getText();
    return std::make_unique<BinaryOpNode>(std::move(left), std::move(right), op);
}

std::unique_ptr<BinaryOpNode> ASTBuilder::visitComparisonExpr(VCalcParser::ComparisonExprContext *ctx) {
    if (ctx->addSubExpr().size() == 1) {
        return visitAddSubExpr(ctx->addSubExpr(0));
    }
    auto left = visitAddSubExpr(ctx->addSubExpr(0));
    auto right = visitAddSubExpr(ctx->addSubExpr(1));
    std::string op = ctx->op->getText();
    return std::make_unique<BinaryOpNode>(std::move(left), std::move(right), op);
}

std::unique_ptr<BinaryOpNode> ASTBuilder::visitAddSubExpr(VCalcParser::AddSubExprContext *ctx) {
    if (ctx->mulDivExpr().size() == 1) {
        return visitMulDivExpr(ctx->mulDivExpr(0));
    }
    auto left = visitMulDivExpr(ctx->mulDivExpr(0));
    auto right = visitMulDivExpr(ctx->mulDivExpr(1));
    std::string op = ctx->op->getText();
    return std::make_unique<BinaryOpNode>(std::move(left), std::move(right), op);
}

std::unique_ptr<BinaryOpNode> ASTBuilder::visitMulDivExpr(VCalcParser::MulDivExprContext *ctx) {
    if (ctx->rangeExpr().size() == 1) {
        return visitRangeExpr(ctx->rangeExpr(0));
    }
    auto left = visitRangeExpr(ctx->rangeExpr(0));
    auto right = visitRangeExpr(ctx->rangeExpr(1));
    std::string op = ctx->op->getText();
    return std::make_unique<BinaryOpNode>(std::move(left), std::move(right), op);
}

std::unique_ptr<RangeNode> ASTBuilder::visitRangeExpr(VCalcParser::RangeExprContext *ctx) {
    if (ctx->indexExpr().size() == 1) {
        auto single = visitIndexExpr(ctx->indexExpr(0));
        return std::make_unique<RangeNode>(std::move(single), nullptr);
    }
    auto start = visitIndexExpr(ctx->indexExpr(0));
    auto end = visitIndexExpr(ctx->indexExpr(1));
    return std::make_unique<RangeNode>(std::move(start), std::move(end));
}

std::unique_ptr<IndexNode> ASTBuilder::visitIndexExpr(VCalcParser::IndexExprContext *ctx) {
    auto array = visit(ctx->expr(0));
    auto index = visit(ctx->expr(1));
    return std::make_unique<IndexNode>(std::move(array), std::move(index));
}

std::unique_ptr<GeneratorNode> ASTBuilder::visitGenerator(VCalcParser::GeneratorContext *ctx) {
    std::string id = ctx->ID()->getText();
    auto dom = visit(ctx->expr(0));
    auto body = visit(ctx->expr(1));
    return std::make_unique<GeneratorNode>(id, std::move(dom), std::move(body));
}

std::unique_ptr<FilterNode> ASTBuilder::visitFilter(VCalcParser::FilterContext *ctx) {
    std::string id = ctx->ID()->getText();
    auto dom = visit(ctx->expr(0));
    auto pred = visit(ctx->expr(1));
    return std::make_unique<FilterNode>(id, std::move(dom), std::move(pred));
}

std::unique_ptr<ExprNode> ASTBuilder::visitAtom(VCalcParser::AtomContext *ctx) {
    if (ctx->INT()) {
        int value = std::stoi(ctx->INT()->getText());
        return std::make_unique<IntNode>(value);
    } else if (ctx->ID()) {
        std::string id = ctx->ID()->getText();
        return std::make_unique<IdNode>(id);
    } else if (ctx->generator()) {
        return visitGenerator(ctx->generator());
    } else if (ctx->filter()) {
        return visitFilter(ctx->filter());
    } else if (ctx->expr()) {
        return visit(ctx->expr());
    }
    return nullptr;
}