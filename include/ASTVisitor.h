#pragma once

// Forward declarations
class FileNode;
class CondNode;
class LoopNode;
class IntDecNode;
class VectorDecNode;
class AssignNode;
class IntNode;
class IdNode;
class BinaryOpNode;
class RangeNode;
class IndexNode;
class GeneratorNode;
class FilterNode;
class PrintNode;

class ASTVisitor {
public:
    virtual void visit(FileNode* node) = 0;
    // statements
    virtual void visit(CondNode* node) = 0;
    virtual void visit(LoopNode* node) = 0;
    virtual void visit(IntDecNode* node) = 0;
    virtual void visit(VectorDecNode* node) = 0;
    virtual void visit(AssignNode* node) = 0;

    // Values
    virtual void visit(IntNode* node) = 0;
    virtual void visit(IdNode* node) = 0;

    // expressions 
    virtual void visit(BinaryOpNode* node) = 0;
    virtual void visit(RangeNode* node) = 0;
    virtual void visit(IndexNode* node) = 0;

    // generators
    virtual void visit(GeneratorNode* node) = 0;
    virtual void visit(FilterNode* node) = 0; 
    
    virtual void visit(PrintNode* node) = 0;

    virtual ~ASTVisitor() = default;
};
