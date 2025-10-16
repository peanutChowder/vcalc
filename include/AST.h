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

const std::string toString(ValueType type) {
    switch (type) {
    case ValueType::UNKNOWN: return "unknown";
    case ValueType::INTEGER: return "int";
    case ValueType::VECTOR:  return "vector";
    }
    return "invalid";
}

// Expression base node
class ExprNode : public ASTNode {
public:
    ExprNode() { type = ValueType::UNKNOWN; }
};

// Integer literal
class IntNode : public ExprNode {
public:
    int value;
    IntNode(int v) : value(v) { type = ValueType::INTEGER; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Identifier
class IdNode : public ExprNode {
public:
    std::string name;
    IdNode(const std::string& i) : name(i) { type = ValueType::UNKNOWN; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Binary operation
class BinaryOpNode : public ExprNode {
public:
    std::shared_ptr<ExprNode> left;
    std::shared_ptr<ExprNode> right;
    std::string op;
    BinaryOpNode(std::shared_ptr<ExprNode> l, std::shared_ptr<ExprNode> r, const std::string& o)
        : left(std::move(l)), right(std::move(r)), op(o) { type = ValueType::UNKNOWN; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Range expression
class RangeNode : public ExprNode {
public:
    std::shared_ptr<ExprNode> start;
    std::shared_ptr<ExprNode> end;
    RangeNode(std::shared_ptr<ExprNode> s, std::shared_ptr<ExprNode> e)
        : start(std::move(s)), end(std::move(e)) { type = ValueType::VECTOR; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Index expression
class IndexNode : public ExprNode {
public:
    std::shared_ptr<ExprNode> array;
    std::shared_ptr<ExprNode> index;
    IndexNode(std::shared_ptr<ExprNode> a, std::shared_ptr<ExprNode> i)
        : array(std::move(a)), index(std::move(i)) { type = ValueType::UNKNOWN; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Generator expression
class GeneratorNode : public ExprNode {
public:
    std::shared_ptr<IdNode> id;
    std::shared_ptr<ExprNode> domain;
    std::shared_ptr<ExprNode> body;
    GeneratorNode(std::shared_ptr<IdNode> i, std::shared_ptr<ExprNode> d, std::shared_ptr<ExprNode> b)
        : id(i), domain(std::move(d)), body(std::move(b)) { type = ValueType::VECTOR; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Filter expression
class FilterNode : public ExprNode {
public:
    std::shared_ptr<IdNode> id;
    std::shared_ptr<ExprNode> domain;
    std::shared_ptr<ExprNode> predicate;
    FilterNode(std::shared_ptr<IdNode> i, std::shared_ptr<ExprNode> d, std::shared_ptr<ExprNode> p)
        : id(i), domain(std::move(d)), predicate(std::move(p)) { type = ValueType::VECTOR; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Integer declaration
class IntDecNode : public ASTNode {
public:
    std::shared_ptr<IdNode> id;
    std::shared_ptr<ExprNode> value;
    IntDecNode(std::shared_ptr<IdNode> i, std::shared_ptr<ExprNode> v)
        : id(i), value(std::move(v)) { type = ValueType::INTEGER; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Vector declaration
class VectorDecNode : public ASTNode {
public:
    std::shared_ptr<IdNode> id;
    std::shared_ptr<ExprNode> vectorValue;
    VectorDecNode(std::shared_ptr<IdNode> i, std::shared_ptr<ExprNode> v)
        : id(i), vectorValue(std::move(v)) { type = ValueType::VECTOR; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Assignment
class AssignNode : public ASTNode {
public:
    std::shared_ptr<IdNode> id;
    std::shared_ptr<ExprNode> value;
    AssignNode(std::shared_ptr<IdNode> i, std::shared_ptr<ExprNode> v)
        : id(i), value(std::move(v)) { type = ValueType::UNKNOWN; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Print statement
class PrintNode : public ASTNode {
public:
    std::shared_ptr<ExprNode> printExpr;
    PrintNode(std::shared_ptr<ExprNode> e)
        : printExpr(std::move(e)) { type = ValueType::UNKNOWN; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Conditional statement
class CondNode : public ASTNode {
public:
    std::shared_ptr<ExprNode> ifCond;
    std::vector<std::shared_ptr<ASTNode>> body;
    CondNode(std::shared_ptr<ExprNode> cond, std::vector<std::shared_ptr<ASTNode>> stmts)
        : ifCond(std::move(cond)), body(std::move(stmts)) { type = ValueType::UNKNOWN; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Loop statement
class LoopNode : public ASTNode {
public:
    std::shared_ptr<ExprNode> loopCond;
    std::vector<std::shared_ptr<ASTNode>> body;
    LoopNode(std::shared_ptr<ExprNode> cond, std::vector<std::shared_ptr<ASTNode>> stmts)
        : loopCond(std::move(cond)), body(std::move(stmts)) { type = ValueType::UNKNOWN; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// File root node
class FileNode : public ASTNode {
public:
    std::vector<std::shared_ptr<ASTNode>> statements;
    FileNode(std::vector<std::shared_ptr<ASTNode>> stmts)
        : statements(std::move(stmts)) { type = ValueType::UNKNOWN; }
    std::string toString() const override { return "FileNode"; }
    void accept(ASTVisitor& visitor) override;
};
