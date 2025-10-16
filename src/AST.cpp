#include "AST.h"
#include <vector>
#include <string>
#include <memory>
#include "AST.h"
#include "ASTVisitor.h"

// ASTNode base implementation 
ASTNode::~ASTNode() = default;

// ExprNode
ExprNode::ExprNode() { type = ValueType::UNKNOWN; }
ExprNode::~ExprNode() = default;

// IntNode
IntNode::IntNode(int v) : value(v) { type = ValueType::INTEGER; }
std::string IntNode::toString() const {
    return std::to_string(value);
}
void IntNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

// IdNode
IdNode::IdNode(const std::string& i) : id(i) { type = ValueType::UNKNOWN; }
std::string IdNode::toString() const {
    return id;
}
void IdNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

// BinaryOpNode
BinaryOpNode::BinaryOpNode(std::shared_ptr<ExprNode> l, std::shared_ptr<ExprNode> r, const std::string& o)
    : left(std::move(l)), right(std::move(r)), op(o) { type = ValueType::UNKNOWN; }
std::string BinaryOpNode::toString() const {
    return left->toString() + " " + op + " " + right->toString();
}
void BinaryOpNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

// RangeNode
RangeNode::RangeNode(std::shared_ptr<ExprNode> s, std::shared_ptr<ExprNode> e)
    : start(std::move(s)), end(std::move(e)) { type = ValueType::VECTOR; }
std::string RangeNode::toString() const {
    return start->toString() + " .. " + end->toString();
}
void RangeNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

// IndexNode
IndexNode::IndexNode(std::shared_ptr<ExprNode> a, std::shared_ptr<ExprNode> i)
    : array(std::move(a)), index(std::move(i)) { type = ValueType::UNKNOWN; }
std::string IndexNode::toString() const {
    return array->toString() + "[" + index->toString() + "]";
}
void IndexNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

// GeneratorNode
GeneratorNode::GeneratorNode(const std::string& i, std::shared_ptr<ExprNode> d, std::shared_ptr<ExprNode> b)
    : id(i), domain(std::move(d)), body(std::move(b)) { type = ValueType::VECTOR; }
std::string GeneratorNode::toString() const {
    return "[ " + id + " in " + domain->toString() + " | " + body->toString() + " ]";
}
void GeneratorNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

// FilterNode
FilterNode::FilterNode(const std::string& i, std::shared_ptr<ExprNode> d, std::shared_ptr<ExprNode> p)
    : id(i), domain(std::move(d)), predicate(std::move(p)) { type = ValueType::VECTOR; }
std::string FilterNode::toString() const {
    return "[ " + id + " in " + domain->toString() + " & " + predicate->toString() + " ]";
}
void FilterNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

// IntDecNode
IntDecNode::IntDecNode(const std::string& i, std::shared_ptr<ExprNode> v)
    : id(i), value(std::move(v)) { type = ValueType::INTEGER; }
std::string IntDecNode::toString() const {
    return "int " + id + " = " + value->toString();
}
void IntDecNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

// VectorDecNode
VectorDecNode::VectorDecNode(const std::string& i, std::shared_ptr<ExprNode> v)
    : id(i), vectorValue(std::move(v)) { type = ValueType::VECTOR; }
std::string VectorDecNode::toString() const {
    return "vector " + id + " = " + vectorValue->toString();
}
void VectorDecNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

// AssignNode
AssignNode::AssignNode(const std::string& i, std::shared_ptr<ExprNode> v)
    : id(i), value(std::move(v)) { type = ValueType::UNKNOWN; }
std::string AssignNode::toString() const {
    return id + " = " + value->toString();
}
void AssignNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

// PrintNode
PrintNode::PrintNode(std::shared_ptr<ExprNode> e)
    : printExpr(std::move(e)) { type = ValueType::UNKNOWN; }
std::string PrintNode::toString() const {
    return "print(" + printExpr->toString() + ")";
}
void PrintNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

// CondNode
CondNode::CondNode(std::shared_ptr<ExprNode> cond, std::vector<std::shared_ptr<ASTNode>> stmts)
    : ifCond(std::move(cond)), body(std::move(stmts)) { type = ValueType::UNKNOWN; }
std::string CondNode::toString() const {
    std::string result = "if (" + ifCond->toString() + ")\n";
    for (const auto& stmt : body) {
        result += "  " + stmt->toString() + "\n";
    }
    result += "fi";
    return result;
}
void CondNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

// LoopNode
LoopNode::LoopNode(std::shared_ptr<ExprNode> cond, std::vector<std::shared_ptr<ASTNode>> stmts)
    : loopCond(std::move(cond)), body(std::move(stmts)) { type = ValueType::UNKNOWN; }
std::string LoopNode::toString() const {
    std::string result = "loop (" + loopCond->toString() + ")\n";
    for (const auto& stat : body) {
        result += "  " + stat->toString() + "\n";
    }
    result += "pool";
    return result;
}
void LoopNode::accept(ASTVisitor& visitor) { visitor.visit(this); }

// FileNode
FileNode::FileNode(std::vector<std::shared_ptr<ASTNode>> stmts)
    : statements(std::move(stmts)) { type = ValueType::UNKNOWN; }
std::string FileNode::toString() const {
    std::string result;
    for (const auto& stmt : statements) {
        result += stmt->toString() + "\n";
    }
    return result;
}
void FileNode::accept(ASTVisitor& visitor) { visitor.visit(this); }







