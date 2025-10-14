#include "SemanticAnalysisVisitor.h"
#include "VCalcParser.h"
#include "Types.h"

#include <any>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

std::any vcalc::SemanticAnalysisVisitor::visitMulDivExpr(vcalc::VCalcParser::MulDivExprContext *ctx) {
  // No multiplication, go to next recursive layer
  if (ctx->op == nullptr) {
    return visit(ctx->rangeExpr()[0]);
  }

  SymbolType currType = std::any_cast<SymbolType>(visit(ctx->rangeExpr()[0]));


  for (size_t i = 1; i < ctx->rangeExpr().size(); i++) {
    // Promote bool to int
    if (currType == SymbolType::Bool) {
      currType = SymbolType::Int;
    }
    if (!(canAssign(SymbolType::Int, currType) || canAssign(SymbolType::Vector, currType))) {
        throw std::runtime_error(
            "Semantic error: invalid operand type in mul/div operation '" + toString(currType) + "'.");
    }

    SymbolType rhsType = std::any_cast<SymbolType>(visit(ctx->rangeExpr()[i]));
    
    std::optional<SymbolType> opResultType = castResult(currType, rhsType);

    if (opResultType == std::nullopt) {
      throw std::runtime_error(
        "Semantic error: invalid operand type in mul/div operation '" + toString(currType) + "'.");
    }
    currType = *opResultType;
  }


  return currType;
}

std::any vcalc::SemanticAnalysisVisitor::visitRangeExpr(vcalc::VCalcParser::RangeExprContext *ctx) {
  SymbolType leftType = std::any_cast<SymbolType>(visit(ctx->indexExpr()[0]));
  
  if (ctx->indexExpr().size() == 1) {
    return leftType;
  }

  SymbolType rightType = std::any_cast<SymbolType>(visit(ctx->indexExpr()[1]));

  if (!(canAssign(SymbolType::Int, leftType) & canAssign(SymbolType::Int, rightType))) {
    throw std::runtime_error("Semantic error: range expression operands must be integers, not" + std::string(toString(leftType)) + " and " + std::string(toString(rightType)) + ".");
  }

  return SymbolType::Vector;

}

std::any vcalc::SemanticAnalysisVisitor::visitIndexExpr(vcalc::VCalcParser::IndexExprContext *ctx) {
  SymbolType domainType = std::any_cast<SymbolType>(visit(ctx->atom()));

  // No indexing operator, fallthrough
  if (ctx->expr() == nullptr) {
    return domainType;
  } 

  if (domainType != SymbolType::Vector) {
    throw std::runtime_error("Semantic error: cannot index non-vector type.");
  }

  SymbolType indexType = std::any_cast<SymbolType>(visit(ctx->expr()));
  if (!(indexType == SymbolType::Int || indexType == SymbolType::Vector)) {
    throw std::runtime_error("Semantic error: cannot index vector with type '" + std::string(toString(indexType)) + "'.");
  }

  if (indexType == SymbolType::Vector) {
    return SymbolType::Vector;
  }

  return SymbolType::Int;
}

std::any vcalc::SemanticAnalysisVisitor::visitAtom(vcalc::VCalcParser::AtomContext *ctx) {
  if (ctx->INT()) {
    semanticAssertValidTypeValue(SymbolType::Int, ctx->INT()->getText());
    return SymbolType::Int;
    
  } else if (ctx->ID()) {
    SymbolInfo* symbol = current_ ? current_->resolve(ctx->ID()->getText()) : nullptr;
    if (symbol == nullptr) {
      throw std::runtime_error("Semantic error: variable " + ctx->ID()->getText() + " used before declaration.");
    }
    return symbol->type;

  } else if (ctx->generator()) {
    return visit(ctx->generator());

  } else if (ctx->filter()) {
    return visit(ctx->filter());

  } else if (ctx->PARENLEFT()) { // handle parenthesized expressions
    return visit(ctx->expr());
  }


  throw std::runtime_error("Semantic error: unsupported atom construct.");
}

void vcalc::SemanticAnalysisVisitor::enterScopeFor(const antlr4::ParserRuleContext* ownerCtx) {
  if (current_ == nullptr) {
    // Should not happen if visitProg initialized current_
    root_ = std::make_unique<Scope>(nullptr);
    current_ = root_.get();
  }
  Scope* child = current_->createChild();
  scopeByCtx_[ownerCtx] = child;
  current_ = child;
}

void vcalc::SemanticAnalysisVisitor::exitScope() {
  if (current_ && current_->parent()) {
    current_ = current_->parent();
  }
}
