#include "AST.h"



AST::AST() {}
std::string AST:toString() const{
    return "AST";
}

// int
void IntNode::accept(ASTVisitor& visitor) {
    visitor.visit(this);
}

// id 
void IdNode::accept(ASTVisitor& visitor){
    visitor.visit(this);
}

// Expression
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
}

//Generator
// [<domain variable> in <domain> | <expression>]
std::string GeneratorNode::toString() const {
    return "[ " getId() + " in " + getDomExpr() + " | " + this.getBodyExpr() + " ]";
}
void GeneratorNode::accept(ASTVisitor& visitor) {
    visitor.visit(this);
}

//Filter
// [<domain variable> in <domain> & <predicate>]
std::string FilterNode::toString() const {
    return "[ " getId() + " in " + getDomExpr() + " & " + this.getPredExpr() + " ]";
}
void FilterNode::accept(ASTVisitor& visitor) {
    visitor.visit(this);
}


