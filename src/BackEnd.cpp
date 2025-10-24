#include <assert.h>

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "BackEnd.h"
// Code generator and lowering class that takes high-level IR and lowers it to
// LLVM IR for further compilation/execution.
BackEnd::BackEnd() : loc(mlir::UnknownLoc::get(&context)) {
    // Load Dialects.
    context.loadDialect<mlir::LLVM::LLVMDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();
    context.loadDialect<mlir::scf::SCFDialect>();
    context.loadDialect<mlir::cf::ControlFlowDialect>();
    context.loadDialect<mlir::memref::MemRefDialect>(); 
    context.loadDialect<mlir::func::FuncDialect>();

    // Initialize the MLIR context 
    builder = std::make_shared<mlir::OpBuilder>(&context);
    module = mlir::ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToStart(module.getBody());

    // Some intial setup to get off the ground 
    setupPrintf();
    createGlobalString("%c\0", "charFormat");
    createGlobalString("%d\0", "intFormat");
    createGlobalString("\n\0", "newline");
}
// Generate main function in MLIR.
int BackEnd::emitModule() {
    // Create a minimal func dialect main entry point. MLIRGen will populate the body.
    auto intType = mlir::IntegerType::get(&context, 32);
    auto funcType = mlir::FunctionType::get(&context, /*inputs=*/llvm::ArrayRef<mlir::Type>{}, /*results=*/mlir::ArrayRef<mlir::Type>{mlir::IntegerType::get(&context, 32)});
    mlir::func::FuncOp mainFunc = builder->create<mlir::func::FuncOp>(loc, "main", funcType);
    mlir::Block *entry = mainFunc.addEntryBlock();
    builder->setInsertionPointToStart(entry);
    return 0;
}
// Use passes to lower high-level dialects (Arith, MemRef, etc.).
int BackEnd::lowerDialects() {
    // Set up the MLIR pass manager to iteratively lower all the Ops
    mlir::PassManager pm(&context);

    // Lower SCF to CF (ControlFlow)
    pm.addPass(mlir::createConvertSCFToCFPass());

    // Lower Func to LLVM (for printi/printc wrappers and calls)
    pm.addPass(mlir::createConvertFuncToLLVMPass());

    // Lower Arith to LLVM
    pm.addPass(mlir::createArithToLLVMConversionPass());

    // Lower MemRef to LLVM
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());

    // Lower CF to LLVM
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());

    // Finalize the conversion to LLVM dialect
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    // Run the passes
    if (mlir::failed(pm.run(module))) {
        llvm::errs() << "Pass pipeline failed\n";
        return 1;
    }
    return 0;
}
// Translate MLIR module to LLVM IR.
void BackEnd::dumpLLVM(std::ostream &os) {  
    // The only remaining dialects in our module after the passes are builtin
    // and LLVM. Setup translation patterns to get them to LLVM IR.
    mlir::registerBuiltinDialectTranslation(context);
    mlir::registerLLVMDialectTranslation(context);
    llvm_module = mlir::translateModuleToLLVMIR(module, llvm_context);

    // Create llvm ostream and dump into the output file
    llvm::raw_os_ostream output(os);
    output << *llvm_module;
}

void BackEnd::finalizeWithReturnZero() {
    // Ensure the 'main' function has a terminating return. Do not rely on
    // the current builder insertion point.
    auto mainFunc = module.lookupSymbol<mlir::func::FuncOp>("main");
    if (!mainFunc)
        return;
    mlir::Block &entry = mainFunc.getBody().front();
    if (!entry.empty() && llvm::isa<mlir::func::ReturnOp>(entry.back()))
        return; // already terminated

    mlir::OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToEnd(&entry);
    auto zero = builder->create<mlir::arith::ConstantIntOp>(loc, 0, builder->getI32Type());
    builder->create<mlir::func::ReturnOp>(builder->getUnknownLoc(), mlir::ValueRange{zero});
}

void BackEnd::setupPrintf() {
    // Create a function declaration for printf, the signature is:
    //   i32 (i8*, ...)
    mlir::Type intType = mlir::IntegerType::get(&context, 32);
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(&context);
    auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(intType, ptrTy,
                                                        /*isVarArg=*/true);

    // Insert the printf function into the body of the parent module.
    builder->create<mlir::LLVM::LLVMFuncOp>(loc, "printf", llvmFnType);
}

void BackEnd::createGlobalString(const char *str, const char *stringName) {

    mlir::Type charType = mlir::IntegerType::get(&context, 8);

    // create string and string type
    auto mlirString = mlir::StringRef(str, strlen(str) + 1);
    auto mlirStringType = mlir::LLVM::LLVMArrayType::get(charType, mlirString.size());

    builder->create<mlir::LLVM::GlobalOp>(loc, mlirStringType, /*isConstant=*/true,
                            mlir::LLVM::Linkage::Internal, stringName,
                            builder->getStringAttr(mlirString), /*alignment=*/0);
    return;
}
