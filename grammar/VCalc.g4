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

//separating expr and atom to control precidence
//allows chaining
// intermediate
atom: INT
    | ID
    | generator
    | filter
    | PARENLEFT expr PARENRIGHT
    | ('-' | '+') atom  // unary operators
    ;

index: atom (SQLEFT expr SQRIGHT)*;

// in order of presidence
// uses left recursion to handle compound exprs
expr: expr MULT expr
    | expr DIV expr
    | expr ADD expr
    | expr MINUS expr
    | expr EQEQ expr
    | expr NEQ expr
    | expr LT expr
    | expr GT expr
    | expr DOTDOT expr //range
    | index
    ;

loopStat: assign
    | cond
    | loop
    | print
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