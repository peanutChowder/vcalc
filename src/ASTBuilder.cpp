#include "ASTBuilder.h"
#include "AST.h"
#include "antlr4-runtime.h"
#include <memory>
#include <stdexcept>

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

std::any ASTBuilder::visitExpr(VCalcParser::ExprContext *ctx) {
    return visit(ctx->equalityExpr());
}

std::any ASTBuilder::visitAssign(VCalcParser::AssignContext *ctx) {
    std::shared_ptr<IdNode> id = std::make_shared<IdNode>(ctx->ID()->getText());
    auto expr = visit(ctx->expr());
    return std::make_shared<AssignNode>(id, std::move(expr));
}

std::any ASTBuilder::visitIntDec(VCalcParser::IntDecContext *ctx) {
    std::shared_ptr<IdNode> id = std::make_shared<IdNode>(ctx->ID()->getText());
    auto expr = visit(ctx->expr());
    return std::make_shared<IntDecNode>(id, std::move(expr));
}

std::any ASTBuilder::visitVectorDec(VCalcParser::VectorDecContext *ctx) {
    std::shared_ptr<IdNode> id = std::make_shared<IdNode>(ctx->ID()->getText());
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
    size_t n = ctx->comparisonExpr().size();
    std::shared_ptr<ExprNode> node = std::any_cast<std::shared_ptr<ExprNode>>(visitComparisonExpr(ctx->comparisonExpr(0)));
    for (size_t i = 1; i < n; ++i) {
        // Find which operator was used at this position (i-1)
        std::string op;
        if (ctx->EQEQ(i-1)) {
            op = ctx->EQEQ(i-1)->getText();
        } else if (ctx->NEQ(i-1)) {
            op = ctx->NEQ(i-1)->getText();
        }
        auto right = std::any_cast<std::shared_ptr<ExprNode>>(visitComparisonExpr(ctx->comparisonExpr(i)));
        node = std::make_shared<BinaryOpNode>(node, right, op);
    }
    return node;
}

std::any ASTBuilder::visitComparisonExpr(VCalcParser::ComparisonExprContext *ctx) {
    size_t n = ctx->addSubExpr().size();
    std::shared_ptr<ExprNode> node = std::any_cast<std::shared_ptr<ExprNode>>(visitAddSubExpr(ctx->addSubExpr(0)));
    for (size_t i = 1; i < n; ++i) {
        std::string op;
        if (ctx->LT(i-1)) {
            op = ctx->LT(i-1)->getText();
        } else if (ctx->GT(i-1)) {
            op = ctx->GT(i-1)->getText();
        }
        auto right = std::any_cast<std::shared_ptr<ExprNode>>(visitAddSubExpr(ctx->addSubExpr(i)));
        node = std::make_shared<BinaryOpNode>(node, right, op);
    }
    return node;
}

std::any ASTBuilder::visitAddSubExpr(VCalcParser::AddSubExprContext *ctx) {
    size_t n = ctx->mulDivExpr().size();
    std::shared_ptr<ExprNode> node = std::any_cast<std::shared_ptr<ExprNode>>(visitMulDivExpr(ctx->mulDivExpr(0)));
    for (size_t i = 1; i < n; ++i) {
        std::string op;
        if (ctx->ADD(i-1)) {
            op = ctx->ADD(i-1)->getText();
        } else if (ctx->MINUS(i-1)) {
            op = ctx->MINUS(i-1)->getText();
        }
        auto right = std::any_cast<std::shared_ptr<ExprNode>>(visitMulDivExpr(ctx->mulDivExpr(i)));
        node = std::make_shared<BinaryOpNode>(node, right, op);
    }
    return node;
}

std::any ASTBuilder::visitMulDivExpr(VCalcParser::MulDivExprContext *ctx) {
    size_t n = ctx->rangeExpr().size();
    std::shared_ptr<ExprNode> node = std::any_cast<std::shared_ptr<ExprNode>>(visitRangeExpr(ctx->rangeExpr(0)));
    for (size_t i = 1; i < n; ++i) {
        std::string op;
        if (ctx->MULT(i-1)) {
            op = ctx->MULT(i-1)->getText();
        } else if (ctx->DIV(i-1)) {
            op = ctx->DIV(i-1)->getText();
        }
        auto right = std::any_cast<std::shared_ptr<ExprNode>>(visitRangeExpr(ctx->rangeExpr(i)));
        node = std::make_shared<BinaryOpNode>(node, right, op);
    }
    return node;
}

std::any ASTBuilder::visitRangeExpr(VCalcParser::RangeExprContext *ctx) {
    if (ctx->indexExpr().size() == 1) {
        // If there's only one indexExpr and no '..', return it directly (do not wrap in RangeNode)
        return visitIndexExpr(ctx->indexExpr(0));
    }
    auto start = visitIndexExpr(ctx->indexExpr(0));
    auto end = visitIndexExpr(ctx->indexExpr(1));
    return std::make_shared<RangeNode>(
        std::any_cast<std::shared_ptr<ExprNode>>(start),
        std::any_cast<std::shared_ptr<ExprNode>>(end)
    );
}

std::any ASTBuilder::visitIndexExpr(VCalcParser::IndexExprContext *ctx) {
    auto array = visit(ctx->atom());
    std::shared_ptr<ExprNode> index = nullptr;
    if (ctx->expr()) {
        index = std::any_cast<std::shared_ptr<ExprNode>>(visit(ctx->expr()));
    }
    return std::make_shared<IndexNode>(
        std::any_cast<std::shared_ptr<ExprNode>>(array),
        index
    );
}

std::any ASTBuilder::visitGenerator(VCalcParser::GeneratorContext *ctx) {
    std::shared_ptr<IdNode> id = std::make_shared<IdNode>(ctx->ID()->getText());
    auto dom = visit(ctx->expr(0));
    auto body = visit(ctx->expr(1));
    return std::make_shared<GeneratorNode>(id, std::move(dom), std::move(body));
}

std::any ASTBuilder::visitFilter(VCalcParser::FilterContext *ctx) {
    std::shared_ptr<IdNode> id = std::make_shared<IdNode>(ctx->ID()->getText());
    auto dom = visit(ctx->expr(0));
    auto pred = visit(ctx->expr(1));
    return std::make_shared<FilterNode>(id, std::move(dom), std::move(pred));
}

std::any ASTBuilder::visitAtom(VCalcParser::AtomContext *ctx) {
    if (ctx->INT()) {
        const std::string literal = ctx->INT()->getText();
        try {
            size_t parsedChars = 0;
            int value = std::stoi(literal, &parsedChars, 10);
            if (parsedChars != literal.size()) {
                throw std::runtime_error("TypeError: Integer literal '" + literal + "' is not a valid signed int.");
            }
            return std::make_shared<IntNode>(value);
        } catch (const std::invalid_argument&) {
            throw std::runtime_error("TypeError: Integer literal '" + literal + "' is not a valid signed int.");
        } catch (const std::out_of_range&) {
            throw std::runtime_error("RangeError: Integer literal '" + literal + "' is out of range for signed int.");
        }
    } else if (ctx->ID()) {
        std::string name = ctx->ID()->getText();
        return std::make_shared<IdNode>(name);
    } else if (ctx->generator()) {
        return visitGenerator(ctx->generator());
    } else if (ctx->filter()) {
        return visitFilter(ctx->filter());
    } else if (ctx->expr()) {
        return visit(ctx->expr());
    }
    return nullptr;
}
