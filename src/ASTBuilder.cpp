#include "ASTBuilder.h"
#include "AST.h"
#include "antlr4-runtime.h"
#include <memory>

using namespace vcalc;
using namespace antlr4;

ASTBuilder::ASTBuilder() {
}

antlrcpp::Any ASTBuilder::visitFile(VCalcParser::FileContext *ctx){
    if(!ctx->stat().empty()){
        antlrcpp::Any result = visit(ctx->stat(0));
        return result;
    }
    return nullptr;
}

antlrcpp::Any ASTBuilder::visitAssign(VCalcParser::AssignContext *ctx) {
    std::string id = ctx->ID()->getText();
    antlrcpp::Any result = visit(ctx->expr());
    ExprNode* ptr = result.as<ExprNode*>();
    std::unique_ptr<ExprNode> expr = std::unique_ptr<ExprNode>(ptr);
    return new AssignNode(id, std::move(expr));
}

antlrcpp::Any ASTBuilder::visitIntDec(VCalcParser::IntDecContext *ctx) {
    std::string id = ctx->ID()->getText();
    antlrcpp::Any result = visit(ctx->expr());
    ExprNode* ptr = result.as<ExprNode*>();
    std::unique_ptr<ExprNode> expr = std::unique_ptr<ExprNode>(ptr);
    return new IntDecNode(id, std::move(expr));
}

antlrcpp::Any ASTBuilder::visitVectorDec(VCalcParser::VectorDecContext *ctx) {
    std::string id = ctx->ID()->getText();
    antlrcpp::Any result = visit(ctx->expr());
    ExprNode* ptr = result.as<ExprNode*>();
    std::unique_ptr<ExprNode> expr = std::unique_ptr<ExprNode>(ptr);
    return new VectorDecNode(id, std::move(expr));
}

antlrcpp::Any ASTBuilder::visitPrint(VCalcParser::PrintContext *ctx) {
    antlrcpp::Any result = visit(ctx->expr());
    ExprNode* ptr = result.as<ExprNode*>();
    std::unique_ptr<ExprNode> expr = std::unique_ptr<ExprNode>(ptr);
    return new PrintNode(std::move(expr));
}

antlrcpp::Any ASTBuilder::visitIndexExpr(VCalcParser::IndexExprContext *ctx) {
    antlrcpp::Any arrayResult = visit(ctx->expr(0));
    ExprNode* arrayPtr = arrayResult.as<ExprNode*>();
    std::unique_ptr<ExprNode> arrayChild = std::unique_ptr<ExprNode>(arrayPtr);

    antlrcpp::Any indexResult = visit(ctx->expr(1));
    ExprNode* indexPtr = indexResult.as<ExprNode*>();
    std::unique_ptr<ExprNode> indexChild = std::unique_ptr<ExprNode>(indexPtr);

    return new IndexNode(std::move(arrayChild), std::move(indexChild));
}

antlrcpp::Any ASTBuilder::visitGenerator(VCalcParser::GeneratorContext *ctx) {
    std::string id = ctx->ID()->getText();
    antlrcpp::Any domResult = visit(ctx->expr(0));
    antlrcpp::Any bodyResult = visit(ctx->expr(1));
    ExprNode* domPtr = domResult.as<ExprNode*>();
    ExprNode* bodyPtr = bodyResult.as<ExprNode*>();
    std::unique_ptr<ExprNode> domChild = std::unique_ptr<ExprNode>(domPtr);
    std::unique_ptr<ExprNode> bodyChild = std::unique_ptr<ExprNode>(bodyPtr);
    return new GeneratorNode(id, std::move(domChild), std::move(bodyChild));
}

antlrcpp::Any ASTBuilder::visitFilter(VCalcParser::FilterContext *ctx) {
    std::string id = ctx->ID()->getText();
    antlrcpp::Any domResult = visit(ctx->expr(0));
    antlrcpp::Any predResult = visit(ctx->expr(1));
    ExprNode* domPtr = domResult.as<ExprNode*>();
    ExprNode* predPtr = predResult.as<ExprNode*>();
    std::unique_ptr<ExprNode> domChild = std::unique_ptr<ExprNode>(domPtr);
    std::unique_ptr<ExprNode> predChild = std::unique_ptr<ExprNode>(predPtr);
    return new FilterNode(id, std::move(domChild), std::move(predChild));
}

antlrcpp::Any ASTBuilder::visitRangeExpr(VCalcParser::RangeExprContext *ctx) {
    if(ctx->indexExpr().size()==1){
        return visit(ctx->indexExpr(0));
    }
    antlrcpp::Any startResult = visit(ctx->indexExpr(0));
    antlrcpp::Any endResult = visit(ctx->indexExpr(1));
    ExprNode* startPtr = startResult.as<ExprNode*>();
    ExprNode* endPtr = endResult.as<ExprNode*>();
    std::unique_ptr<ExprNode> startChild = std::unique_ptr<ExprNode>(startPtr);
    std::unique_ptr<ExprNode> endChild = std::unique_ptr<ExprNode>(endPtr);
    return new RangeNode(std::move(startChild), std::move(endChild));
}

// comparisonExpr (op=(EQEQ|NEQ) comparisonExpr)
antlrcpp::Any ASTBuilder::visitEqualityExpr(VCalcParser::EqualityExprContext *ctx){
    if(ctx->comparisonExpr().size() == 1){
        return visit(ctx->comparisonExpr(0));
    }
    antlrcpp::Any leftResult = visit(ctx->comparisonExpr(0));
    antlrcpp::Any rightResult = visit(ctx->comparisonExpr(1));
    ExprNode* leftPtr = leftResult.as<ExprNode*>();
    ExprNode* rightPtr = rightResult.as<ExprNode*>();
    std::unique_ptr<ExprNode> leftChild = std::unique_ptr<ExprNode>(leftPtr);
    std::unique_ptr<ExprNode> rightChild = std::unique_ptr<ExprNode>(rightPtr);
    std::string op;
    if(ctx->EQEQ()){
        op = "==";
    }else if(ctx->NEQ()){
        op = "!=";
    }
    return new BinaryOpNode(op, std::move(leftChild), std::move(rightChild));
}

// addSubExpr (op=(LT|GT) addSubExpr)*
antlrcpp::Any ASTBuilder::visitComparisonExpr(VCalcParser::ComparisonExprContext *ctx){
    if(ctx->addSubExpr().size()==1){
        return visit(ctx->addSubExpr(0));
    }
    antlrcpp::Any leftResult = visit(ctx->addSubExpr(0));
    antlrcpp::Any rightResult = visit(ctx->addSubExpr(1));
    ExprNode* leftPtr = leftResult.as<ExprNode*>();
    ExprNode* rightPtr = rightResult.as<ExprNode*>();
    std::unique_ptr<ExprNode> leftChild = std::unique_ptr<ExprNode>(leftPtr);
    std::unique_ptr<ExprNode> rightChild = std::unique_ptr<ExprNode>(rightPtr);
    std::string op;
    if(ctx->LT()){
        op = "<";
    }else if(ctx->GT()){
        op = ">";
    }
    return new BinaryOpNode(op, std::move(leftChild), std::move(rightChild));
}

// mulDivExpr (op=(ADD|MINUS) mulDivExpr)*
antlrcpp::Any ASTBuilder::visitAddSubExpr(VCalcParser::AddSubExprContext *ctx){
    if(ctx->mulDivExpr().size()==1){
        return visit(ctx->mulDivExpr(0));
    }
    antlrcpp::Any leftResult = visit(ctx->mulDivExpr(0));
    antlrcpp::Any rightResult = visit(ctx->mulDivExpr(1));
    ExprNode* leftPtr = leftResult.as<ExprNode*>();
    ExprNode* rightPtr = rightResult.as<ExprNode*>();
    std::unique_ptr<ExprNode> leftChild = std::unique_ptr<ExprNode>(leftPtr);
    std::unique_ptr<ExprNode> rightChild = std::unique_ptr<ExprNode>(rightPtr);
    std::string op;
    if(ctx->PLUS()){
        op = "+";
    }else if(ctx->MINUS()){
        op = "-";
    }
    return new BinaryOpNode(op, std::move(leftChild), std::move(rightChild));
}

// rangeExpr (op=(MULT|DIV) rangeExpr)*
antlrcpp::Any ASTBuilder::visitMulDivExpr(VCalcParser::MulDivExprContext *ctx){
    if(ctx->rangeExpr().size()==1){
        return visit(ctx->rangeExpr(0));
    }
    antlrcpp::Any leftResult = visit(ctx->rangeExpr(0));
    antlrcpp::Any rightResult = visit(ctx->rangeExpr(1));
    ExprNode* leftPtr = leftResult.as<ExprNode*>();
    ExprNode* rightPtr = rightResult.as<ExprNode*>();
    std::unique_ptr<ExprNode> leftChild = std::unique_ptr<ExprNode>(leftPtr);
    std::unique_ptr<ExprNode> rightChild = std::unique_ptr<ExprNode>(rightPtr);
    std::string op;
    if(ctx->MULT()){
        op = "*";
    }else if(ctx->DIV()){
        op = "/";
    }
    return new BinaryOpNode(op, std::move(leftChild), std::move(rightChild));
}

antlrcpp::Any ASTBuilder::visitCond(VCalcParser::CondContext *ctx){
    antlrcpp::Any condResult = visit(ctx->expr());
    ExprNode* condPtr = condResult.as<ExprNode*>();
    std::unique_ptr<ExprNode> cond = std::unique_ptr<ExprNode>(condPtr);
    std::vector<std::unique_ptr<AST>> stats;
    for(auto blockStatCtx : ctx->blockStat()){
        antlrcpp::Any statResult = visit(blockStatCtx);
        AST* statPtr = statResult.as<AST*>();
        std::unique_ptr<AST> stat = std::unique_ptr<AST>(statPtr);
        stats.push_back(std::move(stat));
    }
    return new CondNode(std::move(cond), std::move(stats));
}

antlrcpp::Any ASTBuilder::visitLoop(VCalcParser::LoopContext *ctx){
    antlrcpp::Any condResult = visit(ctx->expr());
    ExprNode* condPtr = condResult.as<ExprNode*>();
    std::unique_ptr<ExprNode> cond = std::unique_ptr<ExprNode>(condPtr);
    std::vector<std::unique_ptr<AST>> stats;
    for(auto blockStatCtx : ctx->blockStat()){
        antlrcpp::Any statResult = visit(blockStatCtx);
        AST* statPtr = statResult.as<AST*>();
        std::unique_ptr<AST> stat = std::unique_ptr<AST>(statPtr);
        stats.push_back(std::move(stat));
    }
    return new LoopNode(std::move(cond), std::move(stats));
}

antlrcpp::Any ASTBuilder::visitAtom(VCalcParser::AtomContext *ctx) {
    if (ctx->INT()) {
        int value = std::stoi(ctx->INT()->getText());
        return new IntNode(value);
    }
    else if (ctx->ID()) {
        std::string id = ctx->ID()->getText();
        return new IdNode(id);
    }
    else if (ctx->generator()) {
        return visit(ctx->generator());
    }
    else if (ctx->filter()) {
        return visit(ctx->filter());
    }
    else if (ctx->expr()) {
        return visit(ctx->expr());
    }
    return nullptr;
}