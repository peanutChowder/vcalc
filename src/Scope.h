
#include "Types.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace vcalc {

struct SymbolInfo {
    std::string identifier;
    SymbolType type;
    std::optional<Value> value; // run-time value
};

class Scope {
public:
    explicit Scope(Scope* parent = nullptr);

    bool declare(const std::string& identifier, SymbolType type);

    SymbolInfo* resolve(const std::string& identifier);
    const SymbolInfo* resolve(const std::string& identifier) const;

    void assign(const std::string& identifier, const Value& value);

    Scope* parent() const { return parent_; }
    const std::unordered_map<std::string, SymbolInfo>& symbols() const { return symbols_; }

    // Persistent tree support
    Scope* createChild();
    const std::vector<std::unique_ptr<Scope>>& children() const { return children_; }

    // Diagnostics
    static void printAllScopes(const Scope& root);
    std::string printScope() const;

private:
    std::unordered_map<std::string, SymbolInfo> symbols_;
    Scope* parent_;
    std::vector<std::unique_ptr<Scope>> children_;
};

} 
