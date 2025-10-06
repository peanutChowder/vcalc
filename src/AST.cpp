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
std::string AST:toString() const{
    return "AST";
}

// toString methods
//Generator
// [<domain variable> in <domain> | <expression>]
std::string GeneratorNode::toString() const {
    return "[ " getId() + " in " + getDomExpr()->toString() + " | " + getBodyExpr()->toString() + " ]";
}
//Filter
// [<domain variable> in <domain> & <predicate>]
std::string FilterNode::toString() const {
    return "[ " getId() + " in " + getDomExpr()->toString() + " & " + getPredExpr()->toString() + " ]";
}
// range
// <expr> .. <expr>
std::string RangeNode::toString() const{
    return getStartExpr()->toString() + " .. " + getEndExpr()->toString();
}
std::string BinaryOpNode::toString(){
    return getLeft()->toString() + getOperation() + getRight()->toString();
}
//! might not work if there are multiple indexes
std::string IndexNode::toString(){
    return getArray()->toString() + "[" + getIndex()->toString() + "]"
}
std::string IntDecNode::toString() const{
    return getId() + " = " + getValue()->toString();
}
std::string IntVectorDecNode::toString(){
    return getId() + " = " + getVector()->toString();
}
std::string PrintNode::toString(){
    return "print(" + getPrintExpr()->toString() + ")";
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

//Node specific code for ExprNode
// abstract class
// reference from Language Implementation Patterns
ExprNode::ExprNode() : AST(), evalType(INVALID) {}
std::string ExprNode::toString() { //override
    if (evalType != INVALID) {
        std::string typeString;
        switch (evalType) {
            case SCALAR: typeString = "SCALAR"; break;
            case VECTOR: typeString = "VECTOR"; break;
            default: typeString = "INVALID"; break;
        }
        return AST::toString() + "type=" + typeString;
    }
    return AST::toString();
    // no implementation of accept since this is a abstract class
}






