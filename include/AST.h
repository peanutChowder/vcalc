#pragma once
#include "antlr4-runtime.h"
#include <vector>
#include <string>
#include <memory>

class ASTVisitor;
class ExprNode;
/* AST types
homogenous: single node type for all constructs
heterogenous: different node types 
*/
class AST{
    public:
        AST();
        virtual ~AST() = default;
        virtual std::string toString() const{ return "AST"};
};

//concrete
class IntNode : public ExprNode {  
    private:
        int value; //attribute
    public:
        //constructer
        IntNode(int value):value(val){};
        //getter
        int getValue() const{ return value;}
        void accept(class ASTVisitor& visitor) override;
};

//concrete
class IdNode : public ExprNode { 
    private:
        std::string id; //attribute
    public:
    //constructer
        ID(string id):value(id);
        string getID() const{return id;}
        void accept(class ASTVisitor& visitor) override;
};

//abstract
class ExprNode : public AST{
    public:
        static const int INVALID = 0
        static const int SCALAR = 1;
        static const int VECTOR = 2;
    private:
        int evalType;
    public:
        //constructor
        ExprNode();
        //destructor - gets called automatically when an object is destroyed or out of scope
        virtual ~ExprNode() = default;
        //getter
        int getEvalType() const{return evalType};
        //setter
        void setEvalType(int type){
            evalType = type;
        };
        //Override this
        std::string toString() const override;
        //make class abstract
        // cannot instantiate
        // must be overridden
        virtual void accept(class ASTVisitor& visitor) = 0;
}

// [<domain variable> in <domain> | <expression>]
// paresed as a atom so it inherits from ExprNode
// expr -> index -> atom -> generator
class GeneratorNode: public ExprNode{
    private:
        std::string id;
        std::unique_ptr<ExprNode> domExpr;
        std::unique_ptr<ExprNode> bodyExpr;
    public:
        //constuctor
        GeneratorNode(const std::string& id,
            std::unique_ptr<ExprNode> domain, 
            std::unique_ptr<ExprNode> body)
            : variable(id), 
            domExpr(std::move(domain)), 
            bodyExpr(std::move(body)){}

        std::string getId(){ return id; }
        ExprNode* getDomExpr(){ return domExpr.get();}
        ExprNode* getAndExpr(){ return bodyExpr.get();}
        std::string toString() const override; 
        virtual void accept(class ASTVisitor& visitor) = 0;
}

// [<domain variable> in <domain> & <predicate>]
class FilterNode: public ASTNode{
    private:
        std::string id;
        std::unique_ptr<ExprNode> domExpr;
        std::unique_ptr<ExprNode> predExpr;
    public:
        //constructor
        FilterNode(const std::string& id,
            std::unique_ptr<ExprNode> domain,
            std::unique_ptr<ExprNode> pred)
            : variable(id),
            domExpr(std::move(domain)),
            predExpr(std::move(pred)){}

        std::string getId(){return id;}
        ExprNode* getDomExpr(){ return domExpr.get();}
        ExprNode* getPredExpr(){ return predExpr.get();}
        std::string toString() const override;
        virutal void accept(class ASTVisitor& visitor) = 0;
}
// not implemented
// public class ScalarDecNode: public ASTnode{
//     std::string id;
//     //expr will evaluate to a certain value
//     std::unique_ptr<Expr> value; 
// }
// public class VectorDecNode: public ASTNode{
//     std::string id;
//     std::unique_ptr<Expr> vec;
// }
// public class PrintNode: public ASTNode{
//     std::unique_prt<Expr> printExpr;
// }
public class BinaryOpNode: public ASTNode{
    private:
        std::unique_ptr<Expr> leftExpr;
        std::unique_ptr<Expr> rightExpr; 
        std::string op;
    public:
        //constructor
        BinaryOpNode(const std::string& op,
                std::unique_ptr<ExprNode> left,
                std::unique_ptr<ExprNode> right)
            :leftExpr(std::move(left)), rightExpr(std::move(right)),
            operator_(op){}
        
        ExprNode* getLeft() const {return left.get()};
        EXprNode* getRight() const {return right.get()};
        const std::string& getOperation() const {return operator_;}

        void accept(ASTVisitor& visitor) override;
        

}