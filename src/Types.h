#pragma once

#include <string>
#include <vector>
#include <variant>

// All possible runtime value types
// ------
using Value = std::variant<int>; 


enum class SymbolType {
    Int,
    Bool,
    Vector
};

struct EvalResult {
    SymbolType type;
    Value value;
};

std::string toString(SymbolType type);
int semanticParseIntLiteral(const std::string& literal);
void semanticAssertValidTypeValue(SymbolType type, const std::string& value);
bool canAssign(SymbolType destination, SymbolType source);
SymbolType inferExpressionType(const std::vector<SymbolType>& operands,
                              const std::vector<SymbolType>& acceptableOperandTargets,
                              SymbolType resultType,
                              const std::string& errorContext);
Value coerceValue(const Value& v, SymbolType dst);
