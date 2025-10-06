#include "AST.h"
#include "antlr4-runtime.h"
#include <vector>
#include <string>
#include <memory>

// sets EvalTypes to nodes
// toString methods
// override methods
AST::AST() {}
std::string AST:toString() const{
    return "AST";
}

// Expression
// abstract class
// reference from Language Implementation Patterns
ExprNode::ExprNode() : AST(), evalType(INVALID) {}
std::string ExprNode::toString() { //override
    if (evalType != INVALID) {
        std::string typeString;
        if (evalType == SCALAR){
            typeString = "SCALAR";
        }else if (evalType == VECTOR){
            typeString = "VECTOR";
        }else{
            typeString = "INVALID";
        }
        return AST::toString() + "type=" + typeString;
    }
    return AST::toString();
    // no implementation of accept since this is a abstract class
}
//concrete classes


//Generator
// [<domain variable> in <domain> | <expression>]
std::string GeneratorNode::toString() const {
    return "[ " getId() + " in " + getDomExpr() + " | " + this.getBodyExpr() + " ]";
}
//Filter
// [<domain variable> in <domain> & <predicate>]
std::string FilterNode::toString() const {
    return "[ " getId() + " in " + getDomExpr() + " & " + this.getPredExpr() + " ]";
}
// [<domain variable> in <domain> | <expression>]
std::string RangeNode::toString() cont{
    return "[ " + getId() + ""
}


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


