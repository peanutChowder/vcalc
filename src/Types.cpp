#include "Types.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

std::string toString(SymbolType type) {
    switch (type) {
        case SymbolType::Int:
            return "int";
        case SymbolType::Bool:
            return "bool";
        case SymbolType::Vector:
            return "vector";
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

std::optional<SymbolType> castResult(SymbolType to, SymbolType from) {
    if (to == from) {
        return to;
    }

    switch (to) {
        case SymbolType::Int:
            if (from == SymbolType::Bool) {
                return SymbolType::Int;
            }
            if (from == SymbolType::Vector) {
                return SymbolType::Vector;
            }
            break;
        case SymbolType::Bool:
            if (from == SymbolType::Int) {
                return SymbolType::Bool;
            }
            break;
        case SymbolType::Vector:
            if (from == SymbolType::Int) {
                return SymbolType::Vector;
            }
            if (from == SymbolType::Bool) {
                return SymbolType::Vector;
            }
            break;
        default:
            break;
    }

    return std::nullopt;
}

namespace {

bool isConvertibleTo(SymbolType source, SymbolType target) {
    auto result = castResult(target, source);
    return result.has_value() && *result == target;
}

bool isConvertibleToAny(SymbolType candidate, const std::vector<SymbolType>& targets) {
    return std::any_of(targets.begin(), targets.end(), [&](SymbolType target) {
        return isConvertibleTo(candidate, target);
    });
}

std::string concatenateAllowedTypes(const std::vector<SymbolType>& targets) {
    std::string allowed;
    for (SymbolType type : targets) {
        if (!allowed.empty()) {
            allowed += " or ";
        }
        allowed += toString(type);
    }
    return allowed;
}

} 

bool canAssign(SymbolType destination, SymbolType source) {
    if (destination == source) {
        return true;
    }

    auto result = castResult(destination, source);
    return result.has_value() && *result == destination;
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
