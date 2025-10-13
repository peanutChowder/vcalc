#include "SemanticAnalysisVisitor.h"
#include "VCalcParser.h"
#include "Types.h"

#include <any>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>


// std::any scalc::SemanticAnalysisVisitor::visitStat(scalc::SCalcParser::ProgContext *ctx) {
//   scopeByCtx_.clear(); // Reset in case of multiple visits
//   root_ = std::make_unique<Scope>(nullptr);
//   current_ = root_.get();
//   scopeByCtx_[ctx] = current_;
//   visitChildren(ctx);
//   return {};
// }

// std::any scalc::SemanticAnalysisVisitor::visitLoop(scalc::SCalcParser::LoopContext *ctx) {
//   SymbolType conditionType = std::any_cast<SymbolType>(visit(ctx->expr()));
//   if (!canAssign(SymbolType::Bool, conditionType)) {
//     throw std::runtime_error("Semantic error: loop condition must be convertible to bool.");
//   }

//   enterScopeFor(ctx);
//   for (auto *stmt : ctx->blockStatement()) {
//     visit(stmt);
//   }
//   exitScope();

//   return {};
// }

// std::any scalc::SemanticAnalysisVisitor::visitCondition(scalc::SCalcParser::ConditionContext *ctx) {
//   SymbolType conditionType = std::any_cast<SymbolType>(visit(ctx->expr()));
//   if (!canAssign(SymbolType::Bool, conditionType)) {
//     throw std::runtime_error("Semantic error: condition must be convertible to bool.");
//   }

//   enterScopeFor(ctx);
//   for (auto *stmt : ctx->blockStatement()) {
//     visit(stmt);
//   }
//   exitScope();

//   return {};
// }

// std::any scalc::SemanticAnalysisVisitor::visitExpr(scalc::SCalcParser::ExprContext *ctx) {
//   return visit(ctx->eqExpr());
// }

// std::any scalc::SemanticAnalysisVisitor::visitAssign(scalc::SCalcParser::AssignContext *ctx) {
//   std::string id = ctx->IDENTIFIER()->getText();
//   SymbolInfo* symbol = current_ ? current_->resolve(id) : nullptr;
//   if (symbol == nullptr) {
//     throw std::runtime_error("Semantic error: variable " + id + " used before declaration.");
//   }

//   SymbolType exprType = std::any_cast<SymbolType>(visit(ctx->expr()));
//   if (!canAssign(symbol->type, exprType)) {
//     throw std::runtime_error(
//       "Semantic error: cannot assign value of type '" + std::string(toString(exprType)) +
//       "' to variable '" + id + "' of type '" + std::string(toString(symbol->type)) + "'.");
//   }

//   return {};
// }

// std::any scalc::SemanticAnalysisVisitor::visitDeclr(scalc::SCalcParser::DeclrContext *ctx) {
//   std::string id = ctx->IDENTIFIER()->getText();
//   SymbolType type = parseType(ctx->TYPE()->getText());
//   bool success = current_ && current_->declare(id, type); // Checks if already declared in this scope

//   if (!success) {
//     throw std::runtime_error("Semantic error: variable " + id + " already declared in this scope.");
//   }

//   SymbolType initializerType = std::any_cast<SymbolType>(visit(ctx->expr()));
//   if (!canAssign(type, initializerType)) {
//     throw std::runtime_error(
//       "Semantic error: cannot initialize variable '" + id + "' of type '" + std::string(toString(type)) +
//       "' with expression of type '" + std::string(toString(initializerType)) + "'.");
//   }

//   return {};
// }

// std::any scalc::SemanticAnalysisVisitor::visitMulDiv(scalc::SCalcParser::MulDivContext *ctx) {
//   std::vector<SymbolType> operands;
//   operands.reserve(ctx->right.size() + 1);
//   operands.push_back(std::any_cast<SymbolType>(visit(ctx->left)));

//   for (auto *right : ctx->right) {
//     operands.push_back(std::any_cast<SymbolType>(visit(right)));
//   }

//   SymbolType result = inferExpressionType(operands,
//                                           {SymbolType::Int},
//                                           SymbolType::Int,
//                                           "arithmetic expression");

//   for (std::size_t i = 0; i < ctx->right.size(); ++i) {
//     if (ctx->op[i]->getText() == "/") {
//       auto *rhsAtom = ctx->right[i];
//       if (rhsAtom->INT_LITERAL() && rhsAtom->INT_LITERAL()->getText() == "0") {
//         throw std::runtime_error("Semantic error: division by zero.");
//       }
//     }
//   }

//   return result;
// }

// std::any scalc::SemanticAnalysisVisitor::visitAddSub(scalc::SCalcParser::AddSubContext *ctx) {
//   std::vector<SymbolType> operands;
//   operands.reserve(ctx->right.size() + 1);
//   operands.push_back(std::any_cast<SymbolType>(visit(ctx->left)));

//   for (auto *right : ctx->right) {
//     operands.push_back(std::any_cast<SymbolType>(visit(right)));
//   }

//   SymbolType result = inferExpressionType(operands,
//                                           {SymbolType::Int},
//                                           SymbolType::Int,
//                                           "addition/subtraction expression");

//   return result;
// }

// std::any scalc::SemanticAnalysisVisitor::visitCmp(scalc::SCalcParser::CmpContext *ctx) {
//   std::vector<SymbolType> operands;
//   operands.reserve(ctx->right.size() + 1);
//   operands.push_back(std::any_cast<SymbolType>(visit(ctx->left)));

//   for (auto *right : ctx->right) {
//     operands.push_back(std::any_cast<SymbolType>(visit(right)));
//   }

//   return inferExpressionType(operands,
//                              {SymbolType::Int},
//                              SymbolType::Bool,
//                              "comparison expression");
// }

// std::any scalc::SemanticAnalysisVisitor::visitEqNeq(scalc::SCalcParser::EqNeqContext *ctx) {
//   std::vector<SymbolType> operands;
//   operands.reserve(ctx->right.size() + 1);
//   operands.push_back(std::any_cast<SymbolType>(visit(ctx->left)));

//   for (auto *right : ctx->right) {
//     operands.push_back(std::any_cast<SymbolType>(visit(right)));
//   }

//   return inferExpressionType(operands,
//                              {SymbolType::Int, SymbolType::Bool},
//                              SymbolType::Bool,
//                              "equality expression");
// }

// std::any scalc::SemanticAnalysisVisitor::visitAtom(scalc::SCalcParser::AtomContext *ctx) {
//   if (ctx->IDENTIFIER()) {
//     SymbolInfo* symbol = current_ ? current_->resolve(ctx->IDENTIFIER()->getText()) : nullptr;
//     if (symbol == nullptr) {
//       throw std::runtime_error("Semantic error: variable " + ctx->IDENTIFIER()->getText() + " used before declaration.");
//     }
//     return symbol->type;
//   } else if (ctx->INT_LITERAL()) {
//     semanticAssertValidTypeValue(SymbolType::Int, ctx->INT_LITERAL()->getText());
//     return SymbolType::Int;
//   } else if (ctx->expr()) {
//     return visit(ctx->expr());
//   }

//   throw std::runtime_error("Semantic error: unsupported atom construct.");
// }

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
