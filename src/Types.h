#pragma once

#include <string>
#include <vector>
#include <variant>

// All possible runtime value types
// ------
// - Int and bools stored as long long due to potential expr evaluating to negative.
using Value = std::variant<long long>; 


enum class SymbolType {
    Int,
    Bool
};

struct EvalResult {
    SymbolType type;
    Value value;
};

std::string toString(SymbolType type);
SymbolType parseType(const std::string& typeLiteral);
int semanticParseIntLiteral(const std::string& literal);
int interpretParseIntLiteral(const std::string& literal);
void semanticAssertValidTypeValue(SymbolType type, const std::string& value);
void interpretAssertValidTypeValue(SymbolType type, const std::string& value);
bool canAssign(SymbolType destination, SymbolType source);
SymbolType inferExpressionType(const std::vector<SymbolType>& operands,
                              const std::vector<SymbolType>& acceptableOperandTargets,
                              SymbolType resultType,
                              const std::string& errorContext);
Value coerceValue(const Value& v, SymbolType dst);
