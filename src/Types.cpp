#include "Types.h"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

std::string toString(SymbolType type) {
    switch (type) {
        case SymbolType::Int:
            return "int";
        case SymbolType::Bool:
            return "bool";
        default:
            throw std::runtime_error("Unknown symbol type");
    }
}

int semanticParseIntLiteral(const std::string& literal) {
    try {
        int parsed = std::stoi(literal);
        return static_cast<int>(parsed);
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("Semantic error: integer literal '" + literal + "' is invalid.");
    } catch (const std::out_of_range&) {
        throw std::runtime_error("Semantic error: integer literal '" + literal + "' is out of range.");
    }
}

void semanticAssertValidTypeValue(SymbolType type, const std::string& value) {
    switch (type) {
        case SymbolType::Int:
            semanticParseIntLiteral(value);
            break;
        case SymbolType::Bool: {
            int parsed = semanticParseIntLiteral(value);
            break;
        }
        default:
            throw std::runtime_error("Unknown symbol type");
    }
}


namespace {

const std::map<SymbolType, std::vector<SymbolType>>& implicitCastTable() {
    static const std::map<SymbolType, std::vector<SymbolType>> table = {
        {SymbolType::Int,  {SymbolType::Bool}},
        {SymbolType::Bool, {SymbolType::Int}}
    };
    return table;
}

bool isConvertibleToAny(SymbolType candidate, const std::vector<SymbolType>& targets) {
    return std::any_of(targets.begin(), targets.end(), [&](SymbolType target) {
        return target == candidate || canAssign(target, candidate);
    });
}

std::string concatenateAllowedTypes(const std::vector<SymbolType>& targets) {
    std::string allowed;
    for (std::size_t i = 0; i < targets.size(); ++i) {
        if (!allowed.empty()) {
            allowed += " or ";
        }
        allowed += toString(targets[i]);
    }
    return allowed;
}

} 

bool canAssign(SymbolType destination, SymbolType source) {
    if (destination == source) {
        return true;
    }

    const auto& table = implicitCastTable();
    auto it = table.find(destination);
    if (it == table.end()) {
        return false;
    }

    const auto &allowedSources = it->second;
    return std::find(allowedSources.begin(), allowedSources.end(), source) != allowedSources.end();
}

SymbolType inferExpressionType(const std::vector<SymbolType>& operands,
                              const std::vector<SymbolType>& acceptableOperandTargets,
                              SymbolType resultType,
                              const std::string& errorContext) {
    if (operands.empty()) {
        throw std::runtime_error("Semantic error: " + errorContext + " contains no operands.");
    }

    if (acceptableOperandTargets.empty()) {
        throw std::runtime_error("Semantic error: " + errorContext + " has no acceptable operand types configured.");
    }

    const std::string allowedList = concatenateAllowedTypes(acceptableOperandTargets);

    for (SymbolType operand : operands) {
        if (!isConvertibleToAny(operand, acceptableOperandTargets)) {
            throw std::runtime_error("Semantic error: " + errorContext +
                                     " expects operands convertible to " + allowedList + ".");
        }
    }

    return resultType;
}

Value coerceValue(const Value& v, SymbolType dst) {
    int val = std::get<int>(v);
    switch (dst) {
        case SymbolType::Int:
            return Value{val};
        case SymbolType::Bool:
            // Represent booleans as 0, 1
            return Value{val != 0 ? 1 : 0};
        default:
            throw std::runtime_error("Runtime error: unsupported destination type in coerceValue.");
    }
}
