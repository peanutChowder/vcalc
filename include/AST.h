#pragma once
#include "antlr4-runtime.h"
#include <vector>
#include <string>
#include <memory>

class ASTVisitor; // needs to be implemented

/* 
Header file for AST tree
Filled with constructors, getters, setters
All implementation specific code is in AST.cpp

AST types
homogenous: single node type for all constructs
heterogenous: different node types - this ds 

TODO: implement control flow nodes

*/
class AST{
    public:
        AST();
        virtual ~AST() = default;
        virtual std::string toString() const { return "AST"; }
};

//abstract
class ExprNode: public AST{
    public:
        static const int INVALID = 0;
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
        int getEvalType() const { return evalType; }
        //setter
        void setEvalType(int type) {
            evalType = type;
        }
        //Override this
        std::string toString() const override;
        //make class abstract
        // cannot instantiate
        // must be overridden
        virtual void accept(ASTVisitor& visitor) = 0;
};

//concrete
class IntNode: public ExprNode {  
    private:
        int value; //attribute
    public:
        //constructor
        IntNode(int value) : value(value) {}
        //getter
        int getValue() const { return value; }
        void accept(ASTVisitor& visitor) override;
};

//concrete
class IdNode: public ExprNode { 
    private:
        std::string id; //attribute
    public:
        //constructor
        IdNode(const std::string& id) : id(id) {}
        std::string getID() const { return id; }
        void accept(ASTVisitor& visitor) override;
};

//concrete
class RangeNode: public ExprNode{
    private:
        std::unique_ptr<ExprNode> startExpr;
        std::unique_ptr<ExprNode> endExpr;
    public:
        //constructor
        RangeNode(std::unique_ptr<ExprNode> start, std::unique_ptr<ExprNode> end):
            startExpr(std::move(start)),
            endExpr(std::move(end)) {}
        ExprNode* getStartExpr() const { return startExpr.get(); }
        ExprNode* getEndExpr() const { return endExpr.get(); }
        std::string toString() const override;
        void accept(ASTVisitor& visitor) override;    
};

//concrete
// [<domain variable> in <domain> | <expression>]
class GeneratorNode : public ExprNode{
    private:
        std::string id;
        std::unique_ptr<ExprNode> domExpr;
        std::unique_ptr<ExprNode> bodyExpr;
    public:
        //constructor
        GeneratorNode(const std::string& id,
            std::unique_ptr<ExprNode> domain, 
            std::unique_ptr<ExprNode> body)
            : id(id), 
            domExpr(std::move(domain)), 
            bodyExpr(std::move(body)) {}

        std::string getId() const { return id; }
        ExprNode* getDomExpr() const { return domExpr.get(); }
        ExprNode* getBodyExpr() const { return bodyExpr.get(); }
        std::string toString() const override; 
        void accept(ASTVisitor& visitor) override;
};

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
            predExpr(std::move(pred)) {}

        std::string getId() const { return id; }
        ExprNode* getDomExpr() const { return domExpr.get(); }
        ExprNode* getPredExpr() const { return predExpr.get(); }
        std::string toString() const override;
        void accept(ASTVisitor& visitor) override;
};

// concrete
// this should handle all operations
class BinaryOpNode: public ExprNode{
    private:
        std::unique_ptr<ExprNode> leftExpr;
        std::unique_ptr<ExprNode> rightExpr; 
        std::string op;
    public:
        //constructor
        BinaryOpNode(const std::string& op,
                std::unique_ptr<ExprNode> left,
                std::unique_ptr<ExprNode> right)
            : op(op), leftExpr(std::move(left)), rightExpr(std::move(right)) {}
        
        ExprNode* getLeft() const { return leftExpr.get(); }
        ExprNode* getRight() const { return rightExpr.get(); }
        const std::string& getOperation() const { return op; }
        std::string toString() const override; 
        void accept(ASTVisitor& visitor) override;
};

//concrete
// v[0] or v[0][0]
class IndexNode: public ExprNode{
    private:
        std::unique_ptr<ExprNode> array;
        std::unique_ptr<ExprNode> index;
    public:
        IndexNode(std::unique_ptr<ExprNode> array, std::unique_ptr<ExprNode> index)
            : array(std::move(array)), index(std::move(index)) {}

        ExprNode* getArray() const { return array.get(); }
        ExprNode* getIndex() const { return index.get(); }
        std::string toString() const override; 
        void accept(ASTVisitor& visitor) override;
};

//statement classes
// int id = <expr>
class IntDecNode: public AST{
    private:
        std::string id;
        std::unique_ptr<ExprNode> value;
    public:
        IntDecNode(const std::string& id, std::unique_ptr<ExprNode> value):
            id(id), value(std::move(value)) {}

        const std::string& getId() const { return id; }
        ExprNode* getValue() const { return value.get(); }
        std::string toString() const override; 
        void accept(ASTVisitor& visitor) override;
};

// vector id = <vector>
class VectorDecNode: public AST{
    private:
        std::string id;
        std::unique_ptr<ExprNode> vec;
    public:
        VectorDecNode(const std::string& id, std::unique_ptr<ExprNode> vec):
            id(id), vec(std::move(vec)) {}

        const std::string& getId() const { return id; }
        ExprNode* getVector() const { return vec.get(); }
        std::string toString() const override; 
        void accept(ASTVisitor& visitor) override;
};

class AssignNode: public AST{
    private:
        std::string id;
        std::unique_ptr<ExprNode> value;
    public:
        AssignNode(const std::string& id, std::unique_ptr<ExprNode> value)
            : id(id), value(std::move(value)) {}
        
        const std::string& getId() const { return id; }
        ExprNode* getValue() const { return value.get(); }
        std::string toString() const override;
        void accept(ASTVisitor& visitor) override;
};

// print(<expr>)
class PrintNode: public AST{
    private:
        std::unique_ptr<ExprNode> printExpr;
    public:
        PrintNode(std::unique_ptr<ExprNode> printStat):
            printExpr(std::move(printStat)) {}
        ExprNode* getPrintExpr() const { return printExpr.get(); }
        std::string toString() const override; 
        void accept(ASTVisitor& visitor) override;
};

// if (<expr>) stat* fi
class CondNode: public AST{
    private:
        std::unique_ptr<ExprNode> ifCond;
        std::vector<std::unique_ptr<AST>> body;
    public:
        CondNode(std::unique_ptr<ExprNode> cond,
            std::vector<std::unique_ptr<AST>> stats):
            ifCond(std::move(cond)), body(std::move(stats)) {}
        ExprNode* getCond() const { return ifCond.get(); }
        const std::vector<std::unique_ptr<AST>>& getBody() const { return body; }
        std::string toString() const override;
        void accept(ASTVisitor& visitor) override;
};

// loop (expr) stat* pool
class LoopNode: public AST{
    private:
        std::unique_ptr<ExprNode> loopCond;
        std::vector<std::unique_ptr<AST>> body;
    public:
        LoopNode(std::unique_ptr<ExprNode> cond,
            std::vector<std::unique_ptr<AST>> stats):
            loopCond(std::move(cond)), body(std::move(stats)) {}

        ExprNode* getCond() const { return loopCond.get(); }
        const std::vector<std::unique_ptr<AST>>& getBody() const { return body; }
        std::string toString() const override;
        void accept(ASTVisitor& visitor) override;
};
