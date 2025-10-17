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
#include "mlir/Dialect/Func/IR/FuncOps.h"


#include <stdexcept>

MLIRGen::MLIRGen(BackEnd& backend)
    : backend_(backend),
      builder_(*backend.getBuilder()),
      module_(backend.getModule()),
      context_(backend.getContext()),
      loc_(backend.getLoc()) {}

mlir::Value MLIRGen::popValue() {
    if (v_stack_.empty()) {
        throw std::runtime_error("MLIRGen internal error: value stack underflow.");
    }
    mlir::Value value = v_stack_.back();
    v_stack_.pop_back();
    return value;
}

void MLIRGen::pushValue(mlir::Value value) {
    if (!value) {
        throw std::runtime_error("MLIRGen internal error: attempting to push empty value onto stack.");
    }
    v_stack_.push_back(value);
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
    mlir::Value val = popValue();

    // Integer print
    if (val.getType() == builder_.getI32Type()) {
        // Define function type for printi(i32)
        auto funcType = mlir::FunctionType::get(
            builder_.getContext(),
            {builder_.getI32Type()},
            {}
        );

        // Look up or create the printi function symbol
        auto printFunc = module_.lookupSymbol<mlir::func::FuncOp>("printi");
        if (!printFunc) {
            printFunc = mlir::func::FuncOp::create(loc_, "printi", funcType);
            module_.push_back(printFunc);
        }

        // Call printi(val)
        builder_.create<mlir::func::CallOp>(
            loc_,
            "printi",
            mlir::TypeRange(),
            mlir::ValueRange{val}
        );

        // newline after scalar
        auto printcFunc = module_.lookupSymbol<mlir::func::FuncOp>("printc");
        if (!printcFunc) {
            auto charFuncType = mlir::FunctionType::get(
                builder_.getContext(),
                {builder_.getI8Type()},
                {}
            );
            printcFunc = mlir::func::FuncOp::create(loc_, "printc", charFuncType);
            module_.push_back(printcFunc);
        }

        auto newlineChar = builder_.create<mlir::arith::ConstantIntOp>(loc_, '\n', builder_.getI8Type());
        builder_.create<mlir::func::CallOp>(loc_, "printc", mlir::TypeRange(), mlir::ValueRange{newlineChar});
    }

    
    else if (val.getType().isa<mlir::MemRefType>()) { // Vector print
        auto vecVal = val;
        auto vecSize = builder_.create<mlir::memref::DimOp>(loc_, vecVal, 0);

        auto zeroIdx = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
        auto stepIdx = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);

        // Create helper function types
        auto printiFunc = module_.lookupSymbol<mlir::func::FuncOp>("printi");
        if (!printiFunc) {
            auto funcType = mlir::FunctionType::get(&context_, {builder_.getI32Type()}, {});
            printiFunc = mlir::func::FuncOp::create(loc_, "printi", funcType);
            module_.push_back(printiFunc);
        }

        auto printcFunc = module_.lookupSymbol<mlir::func::FuncOp>("printc");
        if (!printcFunc) {
            auto charFuncType = mlir::FunctionType::get(&context_, {builder_.getI8Type()}, {});
            printcFunc = mlir::func::FuncOp::create(loc_, "printc", charFuncType);
            module_.push_back(printcFunc);
        }

        // Print '['
        auto openBracket = builder_.create<mlir::arith::ConstantIntOp>(loc_, '[', builder_.getI8Type());
        builder_.create<mlir::func::CallOp>(loc_, "printc", mlir::TypeRange(), mlir::ValueRange{openBracket});

        // Loop over elements
        builder_.create<mlir::scf::ForOp>(
            loc_, zeroIdx, vecSize, stepIdx,
            [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc, mlir::Value iv) {
                mlir::Value elem = nestedBuilder.create<mlir::memref::LoadOp>(nestedLoc, vecVal, iv);
                nestedBuilder.create<mlir::func::CallOp>(
                    nestedLoc, "printi", mlir::TypeRange(), mlir::ValueRange{elem});

                // If not last element: print space
                mlir::Value lastIdx = nestedBuilder.create<mlir::arith::SubIOp>(nestedLoc, vecSize, stepIdx);
                mlir::Value isNotLast = nestedBuilder.create<mlir::arith::CmpIOp>(
                    nestedLoc, mlir::arith::CmpIPredicate::ne, iv, lastIdx);

                auto spaceChar = nestedBuilder.create<mlir::arith::ConstantIntOp>(
                    nestedLoc, ' ', nestedBuilder.getI8Type());
                nestedBuilder.create<mlir::scf::IfOp>(
                    nestedLoc, isNotLast,
                    [&](mlir::OpBuilder &ifBuilder, mlir::Location ifLoc) {
                        ifBuilder.create<mlir::func::CallOp>(
                            ifLoc, "printc", mlir::TypeRange(), mlir::ValueRange{spaceChar});
                        ifBuilder.create<mlir::scf::YieldOp>(ifLoc);
                    },
                    nullptr);
            });

        //  closing bracket and newline
        auto closeBracket = builder_.create<mlir::arith::ConstantIntOp>(loc_, ']', builder_.getI8Type());
        builder_.create<mlir::func::CallOp>(loc_, "printc", mlir::TypeRange(), mlir::ValueRange{closeBracket});

        auto newlineChar = builder_.create<mlir::arith::ConstantIntOp>(loc_, '\n', builder_.getI8Type());
        builder_.create<mlir::func::CallOp>(loc_, "printc", mlir::TypeRange(), mlir::ValueRange{newlineChar});
    }
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
    mlir::Value domainVec = v_stack_.back(); v_stack_.pop_back();
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
    auto start = v_stack_.back();
    v_stack_.pop_back();
    node->end->accept(*this);
    auto end = v_stack_.back();
    v_stack_.pop_back();
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

void MLIRGen::visit(IndexNode *node) {
    node->array->accept(*this);
    node->index->accept(*this);

    mlir::Value indexVal = popValue();
    mlir::Value vecVal = popValue();

    auto vecType = vecVal.getType().dyn_cast<mlir::MemRefType>();
    if (!vecType) {
        throw std::runtime_error("MLIRGen error: cannot index non-vector value.");
    }

    if (indexVal.getType().isa<mlir::IntegerType>()) {
        mlir::Value idx = indexVal;
        if (!idx.getType().isa<mlir::IndexType>()) {
            idx = builder_.create<mlir::arith::IndexCastOp>(
                loc_, builder_.getIndexType(), indexVal);
        }

        auto size = builder_.create<mlir::memref::DimOp>(loc_, vecVal, 0);
        auto zeroIdx = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);

        auto isGE = builder_.create<mlir::arith::CmpIOp>(
            loc_, mlir::arith::CmpIPredicate::sge, idx, zeroIdx);
        auto isLT = builder_.create<mlir::arith::CmpIOp>(
            loc_, mlir::arith::CmpIPredicate::slt, idx, size);
        auto inBounds = builder_.create<mlir::arith::AndIOp>(loc_, isGE, isLT);

        // load element and create 0 constant
        auto element = builder_.create<mlir::memref::LoadOp>(loc_, vecVal, idx);
        auto zeroVal = builder_.create<mlir::arith::ConstantIntOp>(loc_, 0, builder_.getI32Type());

        // return 0 if out of bounds
        auto safeElem = builder_.create<mlir::arith::SelectOp>(loc_, inBounds, element, zeroVal);

        pushValue(safeElem);

        } else if (indexVal.getType().isa<mlir::MemRefType>()) { // using a vector as index
            auto indexVec = indexVal;

            auto vecSize = builder_.create<mlir::memref::DimOp>(loc_, vecVal, 0);
            auto indexSize = builder_.create<mlir::memref::DimOp>(loc_, indexVec, 0);

            // Create resulting vector
            auto memrefType = mlir::MemRefType::get({-1}, builder_.getI32Type());
            auto resultVec = builder_.create<mlir::memref::AllocOp>(loc_, memrefType, indexSize);

            auto zeroIdx = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
            auto step = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);

            // For loop: iterate through all indices in indexVec
            auto forOp = builder_.create<mlir::scf::ForOp>(
                loc_, zeroIdx, indexSize, step,
                [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc, mlir::Value iv) {
                    mlir::Value currIdx = nestedBuilder.create<mlir::memref::LoadOp>(nestedLoc, indexVec, iv);
                    // cast
                    if (!currIdx.getType().isa<mlir::IndexType>()) {
                        currIdx = nestedBuilder.create<mlir::arith::IndexCastOp>(nestedLoc, nestedBuilder.getIndexType(), currIdx);
                    }

                    // Within index?
                    auto isGE = nestedBuilder.create<mlir::arith::CmpIOp>(
                        nestedLoc, mlir::arith::CmpIPredicate::sge, currIdx, zeroIdx);
                    auto isLT = nestedBuilder.create<mlir::arith::CmpIOp>(
                        nestedLoc, mlir::arith::CmpIPredicate::slt, currIdx, vecSize);
                    auto inBounds = nestedBuilder.create<mlir::arith::AndIOp>(nestedLoc, isGE, isLT);

                    auto element = nestedBuilder.create<mlir::memref::LoadOp>(nestedLoc, vecVal, currIdx);
                    auto zeroVal = nestedBuilder.create<mlir::arith::ConstantIntOp>(nestedLoc, 0, nestedBuilder.getI32Type());
                    auto safeElem = nestedBuilder.create<mlir::arith::SelectOp>(nestedLoc, inBounds, element, zeroVal);

                    nestedBuilder.create<mlir::memref::StoreOp>(nestedLoc, safeElem, resultVec, iv);
                }
            );

            pushValue(resultVec);
        }
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
    mlir::Value initCondVal = popValue();

    builder_.create<mlir::scf::WhileOp>(
        loc_,
        mlir::TypeRange(builder_.getI32Type()),      
        mlir::ValueRange(initCondVal),                
        [&](mlir::OpBuilder &condBuilder, mlir::Location condLoc, mlir::ValueRange args) {
            auto prevBuilder = builder_;
            builder_ = condBuilder;

            mlir::Value currVal = args[0];

            // Re-evaluate condition expression
            node->loopCond->accept(*this);
            mlir::Value condIntVal = popValue();

            // get truthyness 
            auto zero = condBuilder.create<mlir::arith::ConstantIntOp>(condLoc, 0, condBuilder.getI32Type());
            auto cmpResult = condBuilder.create<mlir::arith::CmpIOp>(
                condLoc, mlir::arith::CmpIPredicate::ne, condIntVal, zero
            );

            // Feed carried variable forward
            condBuilder.create<mlir::scf::ConditionOp>(condLoc, cmpResult, args);

            builder_ = prevBuilder;
        },
        [&](mlir::OpBuilder &bodyBuilder, mlir::Location bodyLoc, mlir::ValueRange args) {
            // Enter body region
            auto prevBuilder = builder_;
            builder_ = bodyBuilder;

            // Visit loop body statements
            for (const auto &stmt : node->body) {
                if (stmt) stmt->accept(*this);
            }

            // At end of body, recompute the condition value for next iteration
            node->loopCond->accept(*this);
            mlir::Value nextCondVal = popValue();

            // Yield the condition value forward
            bodyBuilder.create<mlir::scf::YieldOp>(bodyLoc, nextCondVal);

            builder_ = prevBuilder;
        }
    );
}

