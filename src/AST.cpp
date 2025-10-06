#include "AST.h"
#include "antlr4-runtime.h"
#include <vector>
#include <string>
#include <memory>

/*
sets EvalTypes to nodes
toString methods
override methods
I have evalType as a string rn but we could change it to an enum
Typechecking will take place in the ASTVisitor
*/

AST::AST() {}

std::string AST::toString() const {
    return "AST";
}

//Node specific code for ExprNode
// abstract class
// reference from Language Implementation Patterns
ExprNode::ExprNode() : AST(), evalType(INVALID) {}

std::string ExprNode::toString() const { //override
    if (evalType != INVALID) {
        std::string typeString;
        switch (evalType) {
            case SCALAR: typeString = "SCALAR"; break;
            case VECTOR: typeString = "VECTOR"; break;
            default: typeString = "INVALID"; break;
        }
        return AST::toString() + " type=" + typeString;
    }
    return AST::toString();
    // no implementation of accept since this is an abstract class
}

// toString methods
//Generator
// [<domain variable> in <domain> | <expression>]
std::string GeneratorNode::toString() const {
    return "[ " + getId() + " in " + getDomExpr()->toString() + " | " + getBodyExpr()->toString() + " ]";
}

//Filter
// [<domain variable> in <domain> & <predicate>]
std::string FilterNode::toString() const {
    return "[ " + getId() + " in " + getDomExpr()->toString() + " & " + getPredExpr()->toString() + " ]";
}

// range
// <expr> .. <expr>
std::string RangeNode::toString() const {
    return getStartExpr()->toString() + " .. " + getEndExpr()->toString();
}

std::string BinaryOpNode::toString() const {
    return getLeft()->toString() + " " + getOperation() + " " + getRight()->toString();
}

//! might not work if there are multiple indexes
std::string IndexNode::toString() const {
    return getArray()->toString() + "[" + getIndex()->toString() + "]";
}

std::string IntDecNode::toString() const {
    return "int " + getId() + " = " + getValue()->toString();
}

std::string VectorDecNode::toString() const {
    return "vector " + getId() + " = " + getVector()->toString();
}

std::string AssignNode::toString() const {
    return getId() + " = " + getValue()->toString();
}

std::string PrintNode::toString() const {
    return "print(" + getPrintExpr()->toString() + ")";
}

std::string CondNode::toString() const {
    std::string result = "if (" + ifCond->toString() + ")\n";
    // Print each statement in the body
    for (const auto& stmt : body) {
        result += "  " + stmt->toString() + "\n";  // Indent body statements
    }
    result += "fi";
    return result;
}

std::string LoopNode::toString() const {
    std::string result = "loop (" + loopCond->toString() + ")\n";
    // Print each statement in the body
    for (const auto& stat : body) {
        result += "  " + stat->toString() + "\n";  // Indent body statements
    }
    result += "pool";
    return result;
}

//For visitor
void IntNode::accept(ASTVisitor& visitor) {
    visitor.visit(this);
}

void IdNode::accept(ASTVisitor& visitor) {
    visitor.visit(this);
}

void BinaryOpNode::accept(ASTVisitor& visitor) {
    visitor.visit(this);
}

void RangeNode::accept(ASTVisitor& visitor) {
    visitor.visit(this);
}

void IndexNode::accept(ASTVisitor& visitor) {
    visitor.visit(this);
}

void GeneratorNode::accept(ASTVisitor& visitor) {
    visitor.visit(this);
}

void FilterNode::accept(ASTVisitor& visitor) {
    visitor.visit(this);
}

void IntDecNode::accept(ASTVisitor& visitor) {
    visitor.visit(this);
}

void VectorDecNode::accept(ASTVisitor& visitor) {
    visitor.visit(this);
}

void AssignNode::accept(ASTVisitor& visitor) {
    visitor.visit(this);
}

void PrintNode::accept(ASTVisitor& visitor) {
    visitor.visit(this);
}

void CondNode::accept(ASTVisitor& visitor) {
    visitor.visit(this);
}

void LoopNode::accept(ASTVisitor& visitor) {
    visitor.visit(this);
}







