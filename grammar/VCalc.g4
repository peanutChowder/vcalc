grammar VCalc;
// Tested on lab.antlr.org
program: file;
file: stat* EOF;

stat: intDec END
    | vectorDec END
    | assign END
    | cond END
    | loop END
    | print END
    ;

blockStat
    : assign END
    | cond END
    | loop END
    | print END
    ;

// Expression precedence:
// paren > index > range > mult/div > add/sub > lt/gt > eq/neq
expr
    : equalityExpr
    ;

// equality: ==, !=
equalityExpr
    : comparisonExpr (op=(EQEQ|NEQ) comparisonExpr)*
    ;

// comparison: <, >
comparisonExpr
    : addSubExpr (op=(LT|GT) addSubExpr)*
    ;

// addition and subtraction: +, -
addSubExpr
    : mulDivExpr (op=(ADD|MINUS) mulDivExpr)*
    ;

// multiplication and division: *, /
mulDivExpr
    : rangeExpr (op=(MULT|DIV) rangeExpr)*
    ;

// range and index: .. and []
rangeExpr
    : indexExpr (DOTDOT indexExpr)?
    ;

// index (can chain: a[0][1])
indexExpr
    : atom (SQLEFT expr SQRIGHT)?
    ;

// base atom
atom
    : INT
    | ID
    | generator
    | filter
    | PARENLEFT expr PARENRIGHT
    ;

cond: IF PARENLEFT expr PARENRIGHT blockStat* FI;                      
vectorDec: VECTOR ID EQUAL expr;
intDec: INTKW ID EQUAL expr;
generator: SQLEFT ID IN expr LINE expr SQRIGHT;
filter: SQLEFT ID IN expr AND expr SQRIGHT;
assign: ID EQUAL expr;
loop: LOOP PARENLEFT expr PARENRIGHT blockStat* POOL;              
print: PRINT PARENLEFT expr PARENRIGHT;
// loops && conditionals cannot contain declarations

VECTOR: 'vector';
IF: 'if';
FI: 'fi';
LOOP: 'loop';
POOL: 'pool';
PRINT: 'print';
INTKW: 'int';
IN: 'in';
AND: '&';
LINE: '|';
EQUAL: '=';
END: ';';

SQLEFT: '[';
SQRIGHT: ']';
PARENLEFT: '(';
PARENRIGHT: ')';

DOTDOT: '..';
ADD: '+';
MINUS: '-';
MULT: '*';
DIV: '/';
EQEQ: '==';
NEQ: '!=';
LT: '<';
GT: '>';

INT: [0-9]+;
ID: [a-zA-Z_][a-zA-Z0-9_]*;

COMMENT: '//' ~[\r\n]* -> skip;
WS: [ \t\r\n]+ -> skip;