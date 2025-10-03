grammar VCalc;

//! grammar not tested
// but probably works for now
file: .*? EOF;

stat: intDec END
    | vectorDec END
    | assign END
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
    | '(' expr ')'
    | ('-' | '+') atom  // unary operators
    ; 
index: atom ('[' expr ']')*;
// in order of presidence
// uses left recursion to handle compound exprs
expr: expr ('*' | '/') expr        
    | expr ('+' | '-') expr          
    | expr ('==' | '!=') expr        
    | expr ('<' | '>') expr 
    | expr '..' expr                // 1..5
    | index       
    ;

cond: IF '(' expr ')' stat* FI;
// loops can contain any statement BUT a declaration
loopStat: assign
    | cond
    | loop
    | print
    ;
    
vectorDec: VECTOR ID '=' expr;
intDec: ID '=' expr;
generator: '[' ID IN expr '|' expr ']';
filter: '[' ID IN expr AND expr ']';
assign: ID '=' expr;
loop: LOOP '(' expr ')' loopStat* POOL;
print: PRINT '(' expr ')';
ID: LETTER (LETTER|DIGIT)*;
INT: DIGIT+;
fragment DIGIT: [0-9];
fragment LETTER: [a-zA-Z];
IF: 'if';
FI: 'fi';
LOOP: 'loop';
POOL: 'pool';
PRINT: 'print';
VECTOR: 'vector';
IN: 'in';
AND: '&';
END: ';';
COMMENT: '//' .*? ('\n' | EOF) -> skip ;
// Skip whitespace
WS : [ \t\r\n]+ -> skip ;
