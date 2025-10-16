#pragma once
#include <string>
#include <vector>
#include <memory>
#include "Types.h"

// Forward declaration for visitor
class ASTVisitor;

// Base AST node
class ASTNode {
public:
    ValueType type = ValueType::UNKNOWN;
    virtual ~ASTNode() = default;
    virtual std::string toString() const = 0;
    virtual void accept(ASTVisitor& visitor) = 0;
};

// Expression base node
class ExprNode : public ASTNode {
public:
    ExprNode();
    virtual ~ExprNode() = default;
};

// Integer literal
class IntNode : public ExprNode {
public:
    explicit IntNode(int v);
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;

    int value;
};

// Identifier
class IdNode : public ExprNode {
public:
    explicit IdNode(const std::string& name);
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;

    std::string name;
};

// Binary operation
class BinaryOpNode : public ExprNode {
public:
    BinaryOpNode(std::shared_ptr<ExprNode> left, std::shared_ptr<ExprNode> right, const std::string& op);
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;

    std::shared_ptr<ExprNode> left;
    std::shared_ptr<ExprNode> right;
    std::string op;
};

// Range expression
class RangeNode : public ExprNode {
public:
    RangeNode(std::shared_ptr<ExprNode> start, std::shared_ptr<ExprNode> end);
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;

    std::shared_ptr<ExprNode> start;
    std::shared_ptr<ExprNode> end;
};

// Index expression
class IndexNode : public ExprNode {
public:
    IndexNode(std::shared_ptr<ExprNode> array, std::shared_ptr<ExprNode> index);
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;

    std::shared_ptr<ExprNode> array;
    std::shared_ptr<ExprNode> index;
};

// Generator expression
class GeneratorNode : public ExprNode {
public:
    GeneratorNode(std::shared_ptr<IdNode> id, std::shared_ptr<ExprNode> domain, std::shared_ptr<ExprNode> body);
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;

    std::shared_ptr<IdNode> id;
    std::shared_ptr<ExprNode> domain;
    std::shared_ptr<ExprNode> body;
};

// Filter expression
class FilterNode : public ExprNode {
public:
    FilterNode(std::shared_ptr<IdNode> id, std::shared_ptr<ExprNode> domain, std::shared_ptr<ExprNode> predicate);
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;

    std::shared_ptr<IdNode> id;
    std::shared_ptr<ExprNode> domain;
    std::shared_ptr<ExprNode> predicate;
};

// Integer declaration
class IntDecNode : public ASTNode {
public:
    IntDecNode(std::shared_ptr<IdNode> id, std::shared_ptr<ExprNode> value);
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;

    std::shared_ptr<IdNode> id;
    std::shared_ptr<ExprNode> value;
};

// Vector declaration
class VectorDecNode : public ASTNode {
public:
    VectorDecNode(std::shared_ptr<IdNode> id, std::shared_ptr<ExprNode> vectorValue);
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;

    std::shared_ptr<IdNode> id;
    std::shared_ptr<ExprNode> vectorValue;
};

// Assignment
class AssignNode : public ASTNode {
public:
    AssignNode(std::shared_ptr<IdNode> id, std::shared_ptr<ExprNode> value);
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;

    std::shared_ptr<IdNode> id;
    std::shared_ptr<ExprNode> value;
};

// Print statement
class PrintNode : public ASTNode {
public:
    explicit PrintNode(std::shared_ptr<ExprNode> expr);
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;

    std::shared_ptr<ExprNode> printExpr;
};

// Conditional statement
class CondNode : public ASTNode {
public:
    CondNode(std::shared_ptr<ExprNode> condition, std::vector<std::shared_ptr<ASTNode>> body);
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;

    std::shared_ptr<ExprNode> ifCond;
    std::vector<std::shared_ptr<ASTNode>> body;
};

// Loop statement
class LoopNode : public ASTNode {
public:
    LoopNode(std::shared_ptr<ExprNode> condition, std::vector<std::shared_ptr<ASTNode>> body);
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;

    std::shared_ptr<ExprNode> loopCond;
    std::vector<std::shared_ptr<ASTNode>> body;
};

// File root node
class FileNode : public ASTNode {
public:
    explicit FileNode(std::vector<std::shared_ptr<ASTNode>> statements);
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;

    std::vector<std::shared_ptr<ASTNode>> statements;
};
