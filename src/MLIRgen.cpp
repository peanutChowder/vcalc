/*
Traverse AST tree and for each node emit MLIR operations
Backend sets up MLIR context, builder, and helper functions
After generating the MLIR, Backend will lower the dialects and output LLVM IR
*/
#include "MLIRgen.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include <stdexcept>

MLIRGen::MLIRGen(BackEnd& backend)
    : backend_(backend),
      builder_(*backend.getBuilder()),
      module_(backend.getModule()),
      context_(backend.getContext()),
      loc_(backend.getLoc()) {}

mlir::Value MLIRGen::popValue() {
    if (v_stack.empty()) {
        throw std::runtime_error("MLIRGen internal error: value stack underflow.");
    }
    mlir::Value value = v_stack.back();
    v_stack.pop_back();
    return value;
}

void MLIRGen::pushValue(mlir::Value value) {
    if (!value) {
        throw std::runtime_error("MLIRGen internal error: attempting to push empty value onto stack.");
    }
    v_stack.push_back(value);
}

void MLIRGen::visit(FileNode* node) {
    for (const auto& stmt : node->statements) {
        if (stmt) {
            stmt->accept(*this);
        }
    }
}

void MLIRGen::visit(IntNode* node) {
    auto type = builder_.getI32Type();
    auto attr = builder_.getI32IntegerAttr(node->value);
    auto constant = builder_.create<mlir::arith::ConstantOp>(loc_, type, attr);
    pushValue(constant.getResult());
}

void MLIRGen::visit(IdNode* node) {
    auto it = symbolTable_.find(node->name);
    if (it == symbolTable_.end()) {
        throw std::runtime_error("MLIRGen error: identifier '" + node->name + "' used before assignment.");
    }
    pushValue(it->second);
}

void MLIRGen::visit(IntDecNode* node) {
    node->value->accept(*this);
    mlir::Value value = popValue();
    symbolTable_[node->id->name] = value;
}

void MLIRGen::visit(VectorDecNode* node) {
    node->vectorValue->accept(*this);
    mlir::Value value = popValue();
    symbolTable_[node->id->name] = value;
}

void MLIRGen::visit(AssignNode* node) {
    node->value->accept(*this);
    mlir::Value value = popValue();
    symbolTable_[node->id->name] = value;
}

void MLIRGen::visit(PrintNode* node) {
    node->printExpr->accept(*this);
    popValue();
    // Printing support would lower to a printf call; not implemented yet.
}

//! Add support for different size vectors
void MLIRGen::visit(BinaryOpNode* node) {
    node->left->accept(*this);
    node->right->accept(*this);

    mlir::Value right = popValue();
    mlir::Value left = popValue();

    bool left_isVec = left.getType().isa<mlir::MemRefType>();
    bool right_isVec = right.getType().isa<mlir::MemRefType>();

    mlir::Value result;

      //lambdas for arithmetic and comparison
    auto arithOp = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
        if (node->op == "+") return builder_.create<mlir::arith::AddIOp>(loc_, a, b);
        if (node->op == "-") return builder_.create<mlir::arith::SubIOp>(loc_, a, b);
        if (node->op == "*") return builder_.create<mlir::arith::MulIOp>(loc_, a, b);
        if (node->op == "/") return builder_.create<mlir::arith::DivSIOp>(loc_, a, b);
        throw std::runtime_error("MLIRGen error: unsupported arithmetic operator '" + node->op + "'.");
    };
    auto cmpOp = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
        if (node->op == "<")  return builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::slt, a, b);
        if (node->op == ">")  return builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::sgt, a, b);
        if (node->op == "==") return builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::eq, a, b);
        if (node->op == "!=") return builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::ne, a, b);
        throw std::runtime_error("MLIRGen error: unsupported comparison operator '" + node->op + "'.");
    };
    // if both are integers
    if(!left_isVec && !right_isVec){
        if(node->op == "+" || node->op == "-" || node->op == "*"|| node->op == "/"){
            result = arithOp(left, right);
        }else{
            result = cmpOp(left, right);
        }
        pushValue(result);
        return;
    }
    mlir::Value vec, other;
    bool bothVec;
    if(!left_isVec && right_isVec){
        // right is the vector
        vec = right;
        other = left; // left is scalar
        bothVec = false;
    }else if(!left_isVec && right_isVec){
        vec = left;
        other = right; // right is scalar
        bothVec = false;
    }else{
        vec = right;
        other = left;
        bothVec = true;
    }
    auto size = builder_.create<mlir::memref::DimOp>(loc_, vec, 0);
    auto memrefType = mlir::MemRefType::get({-1}, builder_.getI32Type());
    auto resultVec = builder_.create<mlir::memref::AllocOp>(loc_, memrefType, size);

    auto zero = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
    auto step = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);
    auto forOp = builder_.create<mlir::scf::ForOp>(loc_, zero, size, step);
    builder_.setInsertionPointToStart(forOp.getBody());
    auto iv = forOp.getInductionVar();

    /*
    need to check for same size
    perform operations
    
    
    */
}
    


// domain can be a range or a vector
void MLIRGen::visit(GeneratorNode* node){
    // new scope
    // Scope* savedScope = currentScope;
    // currentScope = currentScope_.createChild();

    // get the full vector if the domain is a range
    node->domain->accept(*this);
    // get domain vector from stack
    mlir::Value domainVec = v_stack.back(); v_stack.pop_back();
    // size = size of domain
    auto size = builder_.create<mlir::memref::DimOp>(loc_, domainVec, 0);

    // allocate space
    auto memrefType = mlir::MemRefType::get({-1}, builder_.getI32Type());
    auto result = builder_.create<mlir::memref::AllocOp>(loc_, memrefType, size);

    // initialize vector 
    auto zero = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
    auto step = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);
    auto forOp = builder_.create<mlir::scf::ForOp>(loc_, zero, size, step);
    builder_.setInsertionPointToStart(forOp.getBody());
    auto iv = forOp.getInductionVar();

    //get domain value
    auto domain_val = builder_.create<mlir::memref::LoadOp>(loc_,domainVec, iv);
    // declare generator var in current scope
    // currentScope->declare(node->id->name, ValueType::INTEGER);
    //place on stack so expression can use it
    pushValue(domain_val);
    // visit expression
    node->body->accept(*this);
    auto bodyResult = popValue();

    // store result in result vector
    builder_.create<mlir::memref::StoreOp>(loc_, bodyResult, result, iv);
    
    // currentScope = savedScope;
    //place result on stack
    pushValue(result);
}

//[<domain variable> in <domain> & <predicate>]
//[i in 1..10 & 5 < i ] == 6 7 8 9 10
void MLIRGen::visit(FilterNode* node){
    // Visit the domain to get the full domain vector
    node->domain->accept(*this);
    mlir::Value domainVec = popValue();
    auto size = builder_.create<mlir::memref::DimOp>(loc_, domainVec, 0);

    // Allocate result vector (max possible size is domain size)
    auto memrefType = mlir::MemRefType::get({-1}, builder_.getI32Type());
    auto result = builder_.create<mlir::memref::AllocOp>(loc_, memrefType, size);

    // Index for writing into result vector
    // keeps track of where we are so we dont have gaps
    auto zeroIdx = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
    auto oneIdx = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);

    // Loop over domain vector
    auto zero = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
    auto step = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);
    auto forOp = builder_.create<mlir::scf::ForOp>(loc_, zero, size, step, mlir::ValueRange{zeroIdx});

    builder_.setInsertionPointToStart(forOp.getBody());
    auto iv = forOp.getInductionVar();
    auto outIdx = forOp.getRegionIterArgs()[0];

    // Load domain value for this index
    auto domainVal = builder_.create<mlir::memref::LoadOp>(loc_, domainVec, iv);

    // Place domain value on stack for predicate
    pushValue(domainVal);

    // Evaluate predicate
    node->predicate->accept(*this);
    auto predResult = popValue();

    // Convert integer predicate to boolean (nonzero = true)
    auto zeroI32 = builder_.create<mlir::arith::ConstantOp>(loc_, builder_.getI32Type(), builder_.getI32IntegerAttr(0));
    auto cond = builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::ne, predResult, zeroI32);

    auto ifOp = builder_.create<mlir::scf::IfOp>(loc_, cond, false);
    // If predicate is true, store domain value in result vector
    builder_.setInsertionPointToStart(&ifOp.getThenRegion().front());
    builder_.create<mlir::memref::StoreOp>(loc_, domainVal, result, outIdx);
    auto incrOutIdx = builder_.create<mlir::arith::AddIOp>(loc_, outIdx, oneIdx);
    builder_.create<mlir::scf::YieldOp>(loc_, incrOutIdx);

    // Else outIdx not incremented
    builder_.setInsertionPointToStart(&ifOp.getElseRegion().front());
    builder_.create<mlir::scf::YieldOp>(loc_, outIdx);

    // make outIdx pass to next iteration
    builder_.setInsertionPointAfter(ifOp);
    builder_.create<mlir::scf::YieldOp>(loc_, ifOp.getResult(0));

    pushValue(result);
}

// x ... x
void MLIRGen::visit(RangeNode* node){
    node->start->accept(*this);
    auto start = v_stack.back();
    v_stack.pop_back();
    node->end->accept(*this);
    auto end = v_stack.back();
    v_stack.pop_back();
    // auto size = (end - start + 1);
    // ie 1 .. 3 = 1 2 3
    auto diff = builder_.create<mlir::arith::SubIOp>(loc_, end, start);
    auto one = builder_.create<mlir::arith::ConstantOp>(loc_, builder_.getI32Type(), builder_.getI32IntegerAttr(1));
    auto size = builder_.create<mlir::arith::AddIOp>(loc_, diff, one);

    //dynamic size vector
    auto memrefType = mlir::MemRefType::get({-1}, builder_.getI32Type());
    auto result = builder_.create<mlir::memref::AllocOp>(loc_, memrefType, size);

    // loop from 0 to size to fill vector
    auto zero = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
    // increment by 1 in the for loop
    auto step = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);
    // for loop
    auto forOp = builder_.create<mlir::scf::ForOp>(loc_, zero, size, step);
    builder_.setInsertionPointToStart(forOp.getBody());
    //loop index
    auto iv = forOp.getInductionVar();
    // value (start + iv)
    auto val = builder_.create<mlir::arith::AddIOp>(loc_, start, iv);
    // store in result vector
    builder_.create<mlir::memref::StoreOp>(loc_, val, result, iv);
    // put result on stack
    pushValue(result);
}

void MLIRGen::visit(CondNode* node) {
    node->ifCond->accept(*this);
    const mlir::Value intVal = popValue();
    auto zero = builder_.create<mlir::arith::ConstantIntOp>(loc_, 0, builder_.getI32Type());
    auto boolVal = builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::ne, intVal, zero);

    builder_.create<mlir::scf::IfOp>(
        loc_,
        boolVal,
        [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc) {
            auto prevBuilder = builder_;
            builder_ = nestedBuilder;

            for (const auto& stmt : node->body) {
                if (stmt) {
                    stmt->accept(*this);
                }
            }

            builder_ = prevBuilder;
        },
        nullptr
    );

}

void MLIRGen::visit(LoopNode *node) {
    node->loopCond->accept(*this);
    const mlir::Value initVal = popValue();
    
    builder_.create<mlir::scf::WhileOp>(
        loc_,
        mlir::TypeRange(builder_.getI32Type()),
        mlir::ValueRange(initVal),
        [&](mlir::OpBuilder &condBuilder, mlir::Location condLoc, mlir::ValueRange args) { 
            auto zero = condBuilder.create<mlir::arith::ConstantIntOp>(condLoc, 0, condBuilder.getI32Type());
            auto cmpResult = condBuilder.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::ne, args[0], zero);

            condBuilder.create<mlir::scf::ConditionOp>(condLoc, cmpResult, args[0]);
        },
        [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc, mlir::ValueRange args) {
            auto prevBuilder = builder_;
            builder_ = nestedBuilder;

            for (const auto &stmt : node->body) {
                if (stmt) stmt->accept(*this);
            }

            nestedBuilder.create<mlir::scf::YieldOp>(loc_);
            builder_ = prevBuilder;
        }
    );
}


