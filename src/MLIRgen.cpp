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
    for (const std::shared_ptr<ASTNode>& stmt : node->statements) {
        if (stmt) {
            stmt->accept(*this);
        }
    }
}

void MLIRGen::visit(IntNode* node) {
    mlir::Type type = builder_.getI32Type();
    mlir::IntegerAttr attr = builder_.getI32IntegerAttr(node->value);
    mlir::arith::ConstantOp constant = builder_.create<mlir::arith::ConstantOp>(loc_, type, attr);
    pushValue(constant.getResult());
}

void MLIRGen::visit(IdNode* node) {
    // If variable is memory-backed, load it
    if (auto mit = varMem_.find(node->name); mit != varMem_.end()) {
        auto mem = mit->second;
        auto zeroIdx = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
        auto val = builder_.create<mlir::memref::LoadOp>(loc_, mem, mlir::ValueRange{zeroIdx});
        pushValue(val);
        return;
    }
    // Fallback to SSA-backed variable
    if (auto it = symbolTable_.find(node->name); it != symbolTable_.end()) {
        pushValue(it->second);
        return;
    }
    throw std::runtime_error("MLIRGen error: identifier '" + node->name + "' used before assignment.");
}

void MLIRGen::visit(IntDecNode* node) {
    node->value->accept(*this);
    mlir::Value value = popValue();
    // allocate memref<1xi32> and store
    auto memTy = mlir::MemRefType::get({1}, builder_.getI32Type());
    auto mem = builder_.create<mlir::memref::AllocOp>(loc_, memTy);
    auto zeroIdx = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
    builder_.create<mlir::memref::StoreOp>(loc_, value, mem, mlir::ValueRange{zeroIdx});
    varMem_[node->id->name] = mem;
}

void MLIRGen::visit(VectorDecNode* node) {
    node->vectorValue->accept(*this);
    mlir::Value value = popValue();
    symbolTable_[node->id->name] = value;
}

void MLIRGen::visit(AssignNode* node) {
    node->value->accept(*this);
    mlir::Value value = popValue();
    auto name = node->id->name;
    if (auto mit = varMem_.find(name); mit != varMem_.end()) {
        auto mem = mit->second;
        auto zeroIdx = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
        builder_.create<mlir::memref::StoreOp>(loc_, value, mem, mlir::ValueRange{zeroIdx});
    } else {
        // create storage if not exists (implicit declaration)
        auto memTy = mlir::MemRefType::get({1}, builder_.getI32Type());
        auto mem = builder_.create<mlir::memref::AllocOp>(loc_, memTy);
        auto zeroIdx = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
        builder_.create<mlir::memref::StoreOp>(loc_, value, mem, mlir::ValueRange{zeroIdx});
        varMem_[name] = mem;
    }
}
void MLIRGen::visit(PrintNode* node) {
    node->printExpr->accept(*this);
    mlir::Value val = popValue();

    // Use LLVM printf via BackEnd globals (intFormat/charFormat/newline)
    auto printfFunc = module_.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf");
    auto intFormat = module_.lookupSymbol<mlir::LLVM::GlobalOp>("intFormat");
    auto charFormat = module_.lookupSymbol<mlir::LLVM::GlobalOp>("charFormat");
    auto newlineStr = module_.lookupSymbol<mlir::LLVM::GlobalOp>("newline");
    if (!printfFunc || !intFormat || !charFormat || !newlineStr) {
        throw std::runtime_error("MLIRGen error: missing printf or format globals in module");
    }

    auto callPrintfFmtOnly = [&](mlir::LLVM::GlobalOp global) {
        auto fmtPtr = builder_.create<mlir::LLVM::AddressOfOp>(loc_, global);
        builder_.create<mlir::LLVM::CallOp>(loc_, printfFunc, mlir::ValueRange{fmtPtr});
    };

    auto callPrintfInt = [&](mlir::Value i32Val) {
        auto fmtPtr = builder_.create<mlir::LLVM::AddressOfOp>(loc_, intFormat);
        builder_.create<mlir::LLVM::CallOp>(loc_, printfFunc, mlir::ValueRange{fmtPtr, i32Val});
    };
    auto callPrintfChar = [&](int ch) {
        auto fmtPtr = builder_.create<mlir::LLVM::AddressOfOp>(loc_, charFormat);
        auto cVal = builder_.create<mlir::arith::ConstantIntOp>(loc_, ch, builder_.getI32Type());
        builder_.create<mlir::LLVM::CallOp>(loc_, printfFunc, mlir::ValueRange{fmtPtr, cVal});
    };

    if (val.getType() == builder_.getI32Type()) {
        callPrintfInt(val);
        callPrintfFmtOnly(newlineStr);
    } else if (val.getType().isa<mlir::MemRefType>()) { // Vector print
        auto vecVal = val;
        auto vecSize = builder_.create<mlir::memref::DimOp>(loc_, vecVal, 0);

        auto zeroIdx = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
        auto stepIdx = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);

        // Print '['
        callPrintfChar('[');

        // Loop over elements
        builder_.create<mlir::scf::ForOp>(
            loc_, zeroIdx, vecSize, stepIdx, mlir::ValueRange(),
            [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc, mlir::Value iv, mlir::ValueRange) {
                mlir::Value elem = nestedBuilder.create<mlir::memref::LoadOp>(nestedLoc, vecVal, iv);
                // print element
                auto fmtPtr = nestedBuilder.create<mlir::LLVM::AddressOfOp>(nestedLoc, intFormat);
                nestedBuilder.create<mlir::LLVM::CallOp>(nestedLoc, printfFunc, mlir::ValueRange{fmtPtr, elem});

                // If not last element: print space
                mlir::Value lastIdx = nestedBuilder.create<mlir::arith::SubIOp>(nestedLoc, vecSize, stepIdx);
                mlir::Value isNotLast = nestedBuilder.create<mlir::arith::CmpIOp>(
                    nestedLoc, mlir::arith::CmpIPredicate::ne, iv, lastIdx);

                auto spaceChar = nestedBuilder.create<mlir::arith::ConstantIntOp>(
                    nestedLoc, ' ', nestedBuilder.getI32Type());
                nestedBuilder.create<mlir::scf::IfOp>(
                    nestedLoc, isNotLast,
                    [&](mlir::OpBuilder &ifBuilder, mlir::Location ifLoc) {
                        auto fmtC = ifBuilder.create<mlir::LLVM::AddressOfOp>(ifLoc, charFormat);
                        ifBuilder.create<mlir::LLVM::CallOp>(ifLoc, printfFunc, mlir::ValueRange{fmtC, spaceChar});
                        ifBuilder.create<mlir::scf::YieldOp>(ifLoc);
                    },
                    [&](mlir::OpBuilder &elseBuilder, mlir::Location elseLoc) {
                        elseBuilder.create<mlir::scf::YieldOp>(elseLoc);
                    });
                
                nestedBuilder.create<mlir::scf::YieldOp>(nestedLoc);
            });

        //  closing bracket and newline
        callPrintfChar(']');
        callPrintfFmtOnly(newlineStr);
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
    auto resultVec = builder_.create<mlir::memref::AllocOp>(loc_, memrefType, mlir::ValueRange{size});

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
    // Evaluate domain to a concrete vector
    node->domain->accept(*this);
    mlir::Value domainVec = popValue();

    // size = size of domain
    auto size = builder_.create<mlir::memref::DimOp>(loc_, domainVec, 0);

    // allocate result vector
    auto memrefType = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, builder_.getI32Type());
    auto result = builder_.create<mlir::memref::AllocOp>(loc_, memrefType, mlir::ValueRange{size});

    // Save any outer binding for the generator variable (SSA symbol table)
    const std::string genName = node->id->name;
    auto prevSym = symbolTable_.find(genName);
    bool hadPrevSym = prevSym != symbolTable_.end();
    mlir::Value prevSymVal;
    if (hadPrevSym) prevSymVal = prevSym->second;

    auto zero = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
    auto step = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);

    // for iv in [0, size):
    builder_.create<mlir::scf::ForOp>(
        loc_, zero, size, step, mlir::ValueRange{},
        [&](mlir::OpBuilder &forBuilder, mlir::Location forLoc, mlir::Value iv, mlir::ValueRange) {
            // i = domainVec[iv]
            auto iVal = forBuilder.create<mlir::memref::LoadOp>(forLoc, domainVec, iv);
            symbolTable_[genName] = iVal;

            // Evaluate body with 'i' in symbol table
            node->body->accept(*this);
            auto elem = popValue();

            // result[iv] = elem
            forBuilder.create<mlir::memref::StoreOp>(forLoc, elem, result, iv);
            forBuilder.create<mlir::scf::YieldOp>(forLoc);
        }
    );

    // Restore outer symbol binding
    if (hadPrevSym) symbolTable_[genName] = prevSymVal; else symbolTable_.erase(genName);

    // Output the result vector
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
    auto memrefType = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, builder_.getI32Type());
    auto result = builder_.create<mlir::memref::AllocOp>(loc_, memrefType, mlir::ValueRange{size});

    // Index for writing into result vector
    // keeps track of where we are so we dont have gaps
    auto zeroIdx = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
    auto oneIdx = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);

    // Loop over domain vector
    auto zero = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
    auto step = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);

    // Conditionally incremented index if predicate evaluates to true
    mlir::Value resultIdx = zero;

    // Save any outer binding for the generator variable (SSA symbol table)
    const std::string genName = node->id->name;
    auto prevSym = symbolTable_.find(genName);
    bool hadPrevSym = prevSym != symbolTable_.end();
    mlir::Value prevSymVal;
    if (hadPrevSym) prevSymVal = prevSym->second;

    // for iv in [0, size):
    builder_.create<mlir::scf::ForOp>(
        loc_, zero, size, step, mlir::ValueRange{},
        [&](mlir::OpBuilder &forBuilder, mlir::Location forLoc, mlir::Value iv, mlir::ValueRange) {
            mlir::OpBuilder outerBuilder = builder_;
            builder_ = forBuilder;

            // i = domainVec[iv], set this before eval predicate
            auto iVal = builder_.create<mlir::memref::LoadOp>(loc_, domainVec, iv);
            // Bind the loop variable as an SSA value (not memory-backed)
            symbolTable_[node->id->name] = iVal;

            // Evaluate predicate
            node->predicate->accept(*this);
            mlir::Value predicateResult = popValue();
            // Convert predicate to i1
            mlir::Value predBool;
            if (auto it = predicateResult.getType().dyn_cast<mlir::IntegerType>()) {
                if (it.getWidth() == 1) {
                    predBool = predicateResult;
                } else {
                    auto zeroI = builder_.create<mlir::arith::ConstantIntOp>(loc_, 0, it);
                    predBool = builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::ne, predicateResult, zeroI);
                }
            } else if (predicateResult.getType().isa<mlir::IndexType>()) {
                auto zeroIdxLocal = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
                predBool = builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::ne, predicateResult, zeroIdxLocal);
            } else {
                throw std::runtime_error("MLIRGen error: unsupported predicate type in filter");
            }

            builder_.create<mlir::scf::IfOp>(
                loc_,
                predBool,
                [&](mlir::OpBuilder &ifBuilder, mlir::Location ifLoc) {
                    // result[resultIdx] = current element from domain
                    ifBuilder.create<mlir::memref::StoreOp>(ifLoc, iVal, result, resultIdx);
                    // advance write index
                    resultIdx = ifBuilder.create<mlir::arith::AddIOp>(ifLoc, resultIdx, step);
                    ifBuilder.create<mlir::scf::YieldOp>(ifLoc);
                }
            );

            // terminate the loop body block
            builder_.create<mlir::scf::YieldOp>(loc_);
            builder_ = outerBuilder;
        }
    );
    // Restore outer symbol binding for loop variable
    if (hadPrevSym) symbolTable_[genName] = prevSymVal; else symbolTable_.erase(genName);

    pushValue(result);
}

// x ... x
void MLIRGen::visit(RangeNode* node){
    node->start->accept(*this);
    auto startIdx = v_stack_.back();
    v_stack_.pop_back();
    node->end->accept(*this);
    auto endIdx = v_stack_.back();
    v_stack_.pop_back();

    if (!startIdx.getType().isa<mlir::IndexType>())
        startIdx = builder_.create<mlir::arith::IndexCastOp>(loc_, builder_.getIndexType(), startIdx);
    if (!endIdx.getType().isa<mlir::IndexType>())
        endIdx = builder_.create<mlir::arith::IndexCastOp>(loc_, builder_.getIndexType(), endIdx);

    mlir::Value diffIdx = builder_.create<mlir::arith::SubIOp>(loc_, endIdx, startIdx);
    mlir::Value oneIdx = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);
    mlir::Value sizeIdx = builder_.create<mlir::arith::AddIOp>(loc_, diffIdx, oneIdx);

    mlir::MemRefType rangeVecTy = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, builder_.getI32Type());



    mlir::Value vector = builder_.create<mlir::memref::AllocOp>(loc_, rangeVecTy, mlir::ValueRange{sizeIdx});
    mlir::Value endPlusOneIdx = builder_.create<mlir::arith::AddIOp>(loc_, endIdx, oneIdx); // Add one for end of loop in for loop

    builder_.create<mlir::scf::ForOp>(
        loc_, 
        startIdx, endPlusOneIdx, oneIdx,
        mlir::ValueRange(),
        [&](mlir::OpBuilder &forBuilder, mlir::Location forLoc, mlir::Value iv, mlir::ValueRange) {
            mlir::Value fromZeroIdx = forBuilder.create<mlir::arith::SubIOp>(forLoc, iv, startIdx);
            mlir::Value valI32 = forBuilder.create<mlir::arith::IndexCastOp>(forLoc, forBuilder.getI32Type(), iv);
            forBuilder.create<mlir::memref::StoreOp>(forLoc, valI32, vector, mlir::ValueRange{fromZeroIdx});

            forBuilder.create<mlir::scf::YieldOp>(forLoc);
        }
    );

    // put result on stack
    pushValue(vector);
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
            auto resultVec = builder_.create<mlir::memref::AllocOp>(loc_, memrefType, mlir::ValueRange{indexSize});

            auto zeroIdx = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
            auto step = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 1);

            // For loop: iterate through all indices in indexVec
            auto forOp = builder_.create<mlir::scf::ForOp>(
                loc_, zeroIdx, indexSize, step, mlir::ValueRange(),
                [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc, mlir::Value iv, mlir::ValueRange) {
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
    mlir::Value condVal = popValue();
    // Coerce to i1
    mlir::Value boolVal;
    if (auto it = condVal.getType().dyn_cast<mlir::IntegerType>()) {
        if (it.getWidth() == 1) {
            boolVal = condVal;
        } else {
            auto zero = builder_.create<mlir::arith::ConstantIntOp>(loc_, 0, it);
            boolVal = builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::ne, condVal, zero);
        }
    } else if (condVal.getType().isa<mlir::IndexType>()) {
        auto zeroIdx = builder_.create<mlir::arith::ConstantIndexOp>(loc_, 0);
        boolVal = builder_.create<mlir::arith::CmpIOp>(loc_, mlir::arith::CmpIPredicate::ne, condVal, zeroIdx);
    } else {
        throw std::runtime_error("MLIRGen error: unsupported condition type");
    }

    builder_.create<mlir::scf::IfOp>(
        loc_,
        boolVal,
        [&](mlir::OpBuilder &thenBuilder, mlir::Location nestedLoc) {
            auto prevBuilder_ = builder_;
            builder_ = thenBuilder;
            for (const auto &st: node->body) {
                st->accept(*this);
            }

            thenBuilder.create<mlir::scf::YieldOp>(nestedLoc);
            builder_ = prevBuilder_;
        }
    );
}

void MLIRGen::visit(LoopNode *node) {
    // while (cond) { body }
    auto toBool = [&](mlir::OpBuilder &bld, mlir::Location loc, mlir::Value v) -> mlir::Value {
        if (auto it = v.getType().dyn_cast<mlir::IntegerType>()) {
            if (it.getWidth() == 1) return v;
            auto zero = bld.create<mlir::arith::ConstantIntOp>(loc, 0, it);
            return bld.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, v, zero);
        } else if (v.getType().isa<mlir::IndexType>()) {
            auto zeroIdx = bld.create<mlir::arith::ConstantIndexOp>(loc, 0);
            return bld.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, v, zeroIdx);
        }
        throw std::runtime_error("MLIRGen error: unsupported loop condition type");
    };

    builder_.create<mlir::scf::WhileOp>(
        loc_, mlir::TypeRange{}, mlir::ValueRange{},
        [&](mlir::OpBuilder &condBuilder, mlir::Location condLoc, mlir::ValueRange) {
            auto prev = builder_;
            builder_ = condBuilder;
            node->loopCond->accept(*this);
            mlir::Value condVal = popValue();
            mlir::Value boolVal = toBool(condBuilder, condLoc, condVal);
            condBuilder.create<mlir::scf::ConditionOp>(condLoc, boolVal, mlir::ValueRange{});
            builder_ = prev;
        },
        [&](mlir::OpBuilder &bodyBuilder, mlir::Location bodyLoc, mlir::ValueRange) {
            auto prev = builder_;
            builder_ = bodyBuilder;
            for (const auto &stmt : node->body) {
                if (stmt) stmt->accept(*this);
            }
            bodyBuilder.create<mlir::scf::YieldOp>(bodyLoc);
            builder_ = prev;
        }
    );
}
