#pragma once
#include <string>
#include <vector>
#include <memory>

// Type enum for all nodes
enum class ValueType {
    UNKNOWN,
    INTEGER,
    VECTOR
};

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
    std::string id;
    IdNode(const std::string& i) : id(i) { type = ValueType::UNKNOWN; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Binary operation
class BinaryOpNode : public ExprNode {
public:
    std::unique_ptr<ExprNode> left;
    std::unique_ptr<ExprNode> right;
    std::string op;
    BinaryOpNode(std::unique_ptr<ExprNode> l, std::unique_ptr<ExprNode> r, const std::string& o)
        : left(std::move(l)), right(std::move(r)), op(o) { type = ValueType::UNKNOWN; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Range expression
class RangeNode : public ExprNode {
public:
    std::unique_ptr<ExprNode> start;
    std::unique_ptr<ExprNode> end;
    RangeNode(std::unique_ptr<ExprNode> s, std::unique_ptr<ExprNode> e)
        : start(std::move(s)), end(std::move(e)) { type = ValueType::VECTOR; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Index expression
class IndexNode : public ExprNode {
public:
    std::unique_ptr<ExprNode> array;
    std::unique_ptr<ExprNode> index;
    IndexNode(std::unique_ptr<ExprNode> a, std::unique_ptr<ExprNode> i)
        : array(std::move(a)), index(std::move(i)) { type = ValueType::UNKNOWN; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Generator expression
class GeneratorNode : public ExprNode {
public:
    std::string id;
    std::unique_ptr<ExprNode> domain;
    std::unique_ptr<ExprNode> body;
    GeneratorNode(const std::string& i, std::unique_ptr<ExprNode> d, std::unique_ptr<ExprNode> b)
        : id(i), domain(std::move(d)), body(std::move(b)) { type = ValueType::VECTOR; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Filter expression
class FilterNode : public ExprNode {
public:
    std::string id;
    std::unique_ptr<ExprNode> domain;
    std::unique_ptr<ExprNode> predicate;
    FilterNode(const std::string& i, std::unique_ptr<ExprNode> d, std::unique_ptr<ExprNode> p)
        : id(i), domain(std::move(d)), predicate(std::move(p)) { type = ValueType::VECTOR; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Integer declaration
class IntDecNode : public ASTNode {
public:
    std::string id;
    std::unique_ptr<ExprNode> value;
    IntDecNode(const std::string& i, std::unique_ptr<ExprNode> v)
        : id(i), value(std::move(v)) { type = ValueType::INTEGER; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Vector declaration
class VectorDecNode : public ASTNode {
public:
    std::string id;
    std::unique_ptr<ExprNode> vectorValue;
    VectorDecNode(const std::string& i, std::unique_ptr<ExprNode> v)
        : id(i), vectorValue(std::move(v)) { type = ValueType::VECTOR; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Assignment
class AssignNode : public ASTNode {
public:
    std::string id;
    std::unique_ptr<ExprNode> value;
    AssignNode(const std::string& i, std::unique_ptr<ExprNode> v)
        : id(i), value(std::move(v)) { type = ValueType::UNKNOWN; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Print statement
class PrintNode : public ASTNode {
public:
    std::unique_ptr<ExprNode> printExpr;
    PrintNode(std::unique_ptr<ExprNode> e)
        : printExpr(std::move(e)) { type = ValueType::UNKNOWN; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Conditional statement
class CondNode : public ASTNode {
public:
    std::unique_ptr<ExprNode> ifCond;
    std::vector<std::unique_ptr<ASTNode>> body;
    CondNode(std::unique_ptr<ExprNode> cond, std::vector<std::unique_ptr<ASTNode>> stmts)
        : ifCond(std::move(cond)), body(std::move(stmts)) { type = ValueType::UNKNOWN; }
    std::string toString() const override;
    void accept(ASTVisitor& visitor) override;
};

// Loop statement
class LoopNode : public ASTNode {
public:
    std::unique_ptr<ExprNode> loopCond;
    std::vector<std::unique_ptr<ASTNode>> body;
    LoopNode(std::unique_ptr<ExprNode> cond, std::vector<std::unique_ptr<ASTNode>> stmts)
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
