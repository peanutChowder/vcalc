#include "ASTBuilder.h"
#include "AST.h"
#include "antlr4-runtime.h"
#include <memory>

using namespace vcalc;
using namespace antlr4;

std::any ASTBuilder::visitFile(VCalcParser::FileContext *ctx) {
    std::vector<std::shared_ptr<ASTNode>> statements;
    for (auto statCtx : ctx->stat()) {
        std::shared_ptr<ASTNode> stmt = std::any_cast<std::shared_ptr<ASTNode>>(visit(statCtx));
        if (stmt) {
            statements.push_back(std::move(stmt));
        }
    }
    return std::make_shared<FileNode>(std::move(statements));
}

std::any ASTBuilder::visitAssign(VCalcParser::AssignContext *ctx) {
    std::string id = ctx->ID()->getText();
    auto expr = visit(ctx->expr());
    return std::make_shared<AssignNode>(id, std::move(expr));
}

std::any ASTBuilder::visitIntDec(VCalcParser::IntDecContext *ctx) {
    std::string id = ctx->ID()->getText();
    auto expr = visit(ctx->expr());
    return std::make_shared<IntDecNode>(id, std::move(expr));
}

std::any ASTBuilder::visitVectorDec(VCalcParser::VectorDecContext *ctx) {
    std::string id = ctx->ID()->getText();
    auto expr = visit(ctx->expr());
    return std::make_shared<VectorDecNode>(id, std::move(expr));
}

std::any ASTBuilder::visitPrint(VCalcParser::PrintContext *ctx) {
    auto expr = visit(ctx->expr());
    return std::make_shared<PrintNode>(std::move(expr));
}

std::any ASTBuilder::visitCond(VCalcParser::CondContext *ctx) {
    auto condExpr = visit(ctx->expr());
    std::vector<std::shared_ptr<ASTNode>> body;
    for (auto statCtx : ctx->blockStat()) {
        std::shared_ptr<ASTNode> stmt = std::any_cast<std::shared_ptr<ASTNode>>(visit(statCtx));
        if (stmt) {
            body.push_back(stmt);
        }
    }
    return std::make_shared<CondNode>(std::move(condExpr), std::move(body));
}

std::any ASTBuilder::visitLoop(VCalcParser::LoopContext *ctx) {
    auto condExpr = visit(ctx->expr());
    std::vector<std::shared_ptr<ASTNode>> body;
    for (auto statCtx : ctx->blockStat()) {
        std::shared_ptr<ASTNode> stmt = std::any_cast<std::shared_ptr<ASTNode>>(visit(statCtx));
        if (stmt) {
            body.push_back(std::move(stmt));
        }
    }
    return std::make_shared<LoopNode>(std::move(condExpr), std::move(body));
}

std::any ASTBuilder::visitEqualityExpr(VCalcParser::EqualityExprContext *ctx) {
    if (ctx->comparisonExpr().size() == 1) {
        return visitComparisonExpr(ctx->comparisonExpr(0));
    }
    auto left = visitComparisonExpr(ctx->comparisonExpr(0));
    auto right = visitComparisonExpr(ctx->comparisonExpr(1));
    std::string op = ctx->op->getText();
    return std::make_shared<BinaryOpNode>(std::move(left), std::move(right), op);
}

std::any ASTBuilder::visitComparisonExpr(VCalcParser::ComparisonExprContext *ctx) {
    if (ctx->addSubExpr().size() == 1) {
        return visitAddSubExpr(ctx->addSubExpr(0));
    }
    auto left = visitAddSubExpr(ctx->addSubExpr(0));
    auto right = visitAddSubExpr(ctx->addSubExpr(1));
    std::string op = ctx->op->getText();
    return std::make_shared<BinaryOpNode>(std::move(left), std::move(right), op);
}

std::any ASTBuilder::visitAddSubExpr(VCalcParser::AddSubExprContext *ctx) {
    if (ctx->mulDivExpr().size() == 1) {
        return visitMulDivExpr(ctx->mulDivExpr(0));
    }
    auto left = visitMulDivExpr(ctx->mulDivExpr(0));
    auto right = visitMulDivExpr(ctx->mulDivExpr(1));
    std::string op = ctx->op->getText();
    return std::make_shared<BinaryOpNode>(std::move(left), std::move(right), op);
}

std::any ASTBuilder::visitMulDivExpr(VCalcParser::MulDivExprContext *ctx) {
    if (ctx->rangeExpr().size() == 1) {
        return visitRangeExpr(ctx->rangeExpr(0));
    }
    auto left = visitRangeExpr(ctx->rangeExpr(0));
    auto right = visitRangeExpr(ctx->rangeExpr(1));
    std::string op = ctx->op->getText();
    return std::make_shared<BinaryOpNode>(std::move(left), std::move(right), op);
}

std::any ASTBuilder::visitRangeExpr(VCalcParser::RangeExprContext *ctx) {
    if (ctx->indexExpr().size() == 1) {
        auto single = visitIndexExpr(ctx->indexExpr(0));
        return std::make_shared<RangeNode>(std::move(single), nullptr);
    }
    auto start = visitIndexExpr(ctx->indexExpr(0));
    auto end = visitIndexExpr(ctx->indexExpr(1));
    return std::make_shared<RangeNode>(std::move(start), std::move(end));
}

std::any ASTBuilder::visitIndexExpr(VCalcParser::IndexExprContext *ctx) {
    auto array = visit(ctx->atom());
    auto index = visit(ctx->expr());
    return std::make_shared<IndexNode>(std::move(array), std::move(index));
}

std::any ASTBuilder::visitGenerator(VCalcParser::GeneratorContext *ctx) {
    std::string id = ctx->ID()->getText();
    auto dom = visit(ctx->expr(0));
    auto body = visit(ctx->expr(1));
    return std::make_shared<GeneratorNode>(id, std::move(dom), std::move(body));
}

std::any ASTBuilder::visitFilter(VCalcParser::FilterContext *ctx) {
    std::string id = ctx->ID()->getText();
    auto dom = visit(ctx->expr(0));
    auto pred = visit(ctx->expr(1));
    return std::make_shared<FilterNode>(id, std::move(dom), std::move(pred));
}

std::any ASTBuilder::visitAtom(VCalcParser::AtomContext *ctx) {
    if (ctx->INT()) {
        int value = std::stoi(ctx->INT()->getText());
        return std::make_shared<IntNode>(value);
    } else if (ctx->ID()) {
        std::string id = ctx->ID()->getText();
        return std::make_shared<IdNode>(id);
    } else if (ctx->generator()) {
        return visitGenerator(ctx->generator());
    } else if (ctx->filter()) {
        return visitFilter(ctx->filter());
    } else if (ctx->expr()) {
        return visit(ctx->expr());
    }
    return nullptr;
}