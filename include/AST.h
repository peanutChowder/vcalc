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

//abstract
class ExprNode: public AST{
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

//concrete
class IntNode: public ExprNode {  
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
class IdNode: public ExprNode { 
    private:
        std::string id; //attribute
    public:
    //constructer
        ID(string id):value(id);
        string getID() const{return id;}
        void accept(class ASTVisitor& visitor) override;
};

//concrete
class RangeNode: public ExprNode{
    private:
        std::unique_ptr<ExprNode> startExpr;
        std::unique_ptr<ExprNode> endExpr;
    public:
        //constructor
        RangeNode(std::unique_ptr<ExprNode> start, std::unqiue_ptr<ExprNode> end):
            startExpr(std::move(start)),
            endExpr(std::move(end)){}
        ExprNode* getStartExpr() const {return startExpr.get();}
        ExprNode* getEndExpr() const {return endExpr.get();}
        std::string toString() const override;
        void accept(ASTVisitor& visitor) override;    
}

//concrete
// [<domain variable> in <domain> | <expression>]
class GeneratorNode ExprNode{
    private:
        std::string id;
        std::unique_ptr<ExprNode> domExpr;
        std::unique_ptr<ExprNode> bodyExpr;
    public:
        //constuctor
        GeneratorNode(const std::string& id,
            std::unique_ptr<ExprNode> domain, 
            std::unique_ptr<ExprNode> body)
            : id(id), 
            domExpr(std::move(domain)), 
            bodyExpr(std::move(body)){}

        std::string getId(){ return id; }
        ExprNode* getDomExpr() const { return domExpr.get();}
        ExprNode* getAndExpr() const { return bodyExpr.get();}
        std::string toString() const override; 
        void accept(class ASTVisitor& visitor) override;
}

//concrete
// [<domain variable> in <domain> & <predicate>]
class FilterNode: public ExprNode{
    private:
        std::string id;
        std::unique_ptr<ExprNode> domExpr;
        std::unique_ptr<ExprNode> predExpr;
    public:
        //constructor
        FilterNode(const std::string& id,
            std::unique_ptr<ExprNode> domain,
            std::unique_ptr<ExprNode> pred)
            : id(id),
            domExpr(std::move(domain)),
            predExpr(std::move(pred)){}

        std::string getId(){return id;}
        ExprNode* getDomExpr() const { return domExpr.get();}
        ExprNode* getPredExpr() const { return predExpr.get();}
        std::string toString() const override;
        void accept(class ASTVisitor& visitor) override;
}

// concrete
// this should handle all operations
class BinaryOpNode: public ExprNode{
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
        
        ExprNode* getLeft() const {return left.get();}
        EXprNode* getRight() const {return right.get();}
        const std::string& getOperation() const {return operator_;}

        void accept(ASTVisitor& visitor) override;
}

//concrete
// v[0] or v[0][0]
class IndexNode: public ExprNode{
    private:
        std::unique_ptr<ExprNode> array;
        std::unique_ptr<ExprNode> index;
    public:
        indexNode(std::unique_ptr<ExprNode> array, std::unique_ptr<ExprNode>index)
        :array(std::move(array)), index(std::move(index)){}

        ExprNode* getArray() const {return array.get();}
        ExprNode* getIndex() const {return index.get();}

        void accept(ASTVisitor& visitor) override;
};

//statement classes

// int id = <expr>
class IntDecNode: public AST{
    private:
        std::string id;
        std::unique_ptr<ExprNode> value;
    public:
        intDecNode(const std::string& id, std::unique_ptr<ExprNode> value):
            id(id) , value(std::move(value)){}

        const std::string& getId() const {return id.get();}
        ExprNode* getValue() const {return value.get();}
        void accept(ASTVisitor& visitor) override;
}
// vector id = <vector>
class VectorDecNode: public AST{
    private:
        std::string id;
        std::unique_ptr<Expr> vec;
    public:
        intDecNode(const std::string& id, std::unique_ptr<ExprNode> vec):
            id(id) , value(std::move(value)){}

        const std::string& getId() const {return id.get();}
        ExprNode* getVector() const {return vec.get();}
        void accept(ASTVisitor& visitor) override;
}
class AssignNode: public AST{
    private: 
}
// print(<expr>)
class PrintNode: public AST{
    private:
        std::unique_prt<Expr> printExpr;
    public:
        printNode(std::unique_ptr<ExprNode> printStat):
            printExpr(std::move(printStat)){}
        ExprNode* getPrintExpr() const {return printExpr.get()}
        void accept(ASTVisitor& visitor) override;

}