#include "VCalcLexer.h"
#include "VCalcParser.h"

#include "ANTLRFileStream.h"
#include "CommonTokenStream.h"
#include "tree/ParseTree.h"
#include "tree/ParseTreeWalker.h"

#include "BackEnd.h"
#include "ASTBuilder.h"
#include "AST.h"

#include <iostream>
#include <fstream>
#include <any>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout << "Missing required argument.\n"
              << "Required arguments: <input file path> <output file path>\n";
    return 1;
  }

  // Open the file then parse and lex it.
  antlr4::ANTLRFileStream afs;
  afs.loadFromFile(argv[1]);
  vcalc::VCalcLexer lexer(&afs);
  antlr4::CommonTokenStream tokens(&lexer);
  vcalc::VCalcParser parser(&tokens);

  // Get the root of the parse tree. Use your base rule name.
  auto fileContext = parser.file();
  ASTBuilder builder;
  std::any astAny = builder.visitFile(fileContext);
  auto ast = std::any_cast<std::shared_ptr<FileNode>>(astAny);

  // HOW TO USE A VISITOR
  // Make the visitor
  // MyVisitor visitor;
  // Visit the tree
  // visitor.visit(tree);

  std::ofstream os(argv[2]);
  BackEnd backend;
  backend.emitModule();
  backend.lowerDialects();
  backend.dumpLLVM(os);

  return 0;
}
