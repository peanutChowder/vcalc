#include "Types.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

std::string toString(ValueType type) {
    switch (type) {
        case ValueType::UNKNOWN:
            return "unknown";
        case ValueType::INTEGER:
            return "int";
        case ValueType::VECTOR:
            return "vector";
    }
    return "invalid";
}
