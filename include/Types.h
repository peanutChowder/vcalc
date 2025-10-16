#pragma once

#include <string>
#include <vector>
#include <variant>

// Type enum for all nodes
enum class ValueType {
    UNKNOWN,
    INTEGER,
    VECTOR
};

const std::string toString(ValueType type) {
    switch (type) {
    case ValueType::UNKNOWN: return "unknown";
    case ValueType::INTEGER: return "int";
    case ValueType::VECTOR:  return "vector";
    }
    return "invalid";
}

inline ValueType promote(ValueType from, ValueType to) {
    switch (from) {
        case ValueType::UNKNOWN:
            return ValueType::UNKNOWN;
            break;
        case ValueType::INTEGER:
            switch (to) {
                case ValueType::UNKNOWN: return ValueType::UNKNOWN;
                case ValueType::INTEGER: return ValueType::INTEGER;
                case ValueType::VECTOR:  return ValueType::VECTOR;
            }
            break;
        case ValueType::VECTOR:
            switch (to) {
                case ValueType::UNKNOWN: return ValueType::UNKNOWN;
                case ValueType::INTEGER: return ValueType::UNKNOWN;
                case ValueType::VECTOR:  return ValueType::VECTOR;
            }
            break;
    }
    return ValueType::UNKNOWN;
}
