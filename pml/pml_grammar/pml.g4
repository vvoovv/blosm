// This is a grammar fragment for PML style language
// Version 03

grammar pml;

styles
    : named_block+  EOF            #NAMED
    | elements EOF                 #UNNAMED
    ;

named_block
    : ('@name' STRING_LITERAL SEMI elements)
    ;

elements
    : element ( element )* 
    ;

element 
    : '@'? 'level' (STRUDEL def_name)? spec_conditions (LPAREN condition RPAREN)? LCURLY attributes RCURLY  
    | '@'? element_name (STRUDEL def_name)? (LPAREN condition RPAREN)? LCURLY attributes RCURLY             
    ;

attributes
    : attribute*
    ;

attribute 
    : 'symmetry' COLON sym_expression SEMI
    | 'use' COLON use_expression SEMI
    | ('faces' | 'sharpEdges') COLON smooth_expression SEMI
    | attr_name COLON expression SEMI
    | attr_name COLON markup_block  // markup
    ;

sym_expression
    : IDENTIFIER
    ;

use_expression
    : IDENTIFIER (COMMA IDENTIFIER)*
    ;

smooth_expression
    : expression
    ;

markup_block
    : LBRACK elements? RBRACK
    ;

expression
    :  simple_expr | function | alternatives
    ;

alternatives
    : function (PIPE function)+
    ;

function
    : 'attr' LPAREN string_literal RPAREN                       #ATTR
    | 'bldgAttr' LPAREN string_literal RPAREN                   #BUILDATTR
    | 'random_normal' LPAREN NUMBER RPAREN                      #RANDN
    | 'random_weighted' nested_list                             #RANDW
    | 'if' LPAREN conditional RPAREN (function )                #COND
    | 'use_from' LPAREN IDENTIFIER RPAREN                       #USEFROM
    | 'per_building' LPAREN (function | alternatives) RPAREN    #PERBUILD
    | 'rgb' LPAREN NUMBER COMMA NUMBER COMMA NUMBER RPAREN      #RGB
    | 'rgba' LPAREN NUMBER COMMA NUMBER COMMA NUMBER COMMA NUMBER RPAREN      #RGBA
    | constant                                                  #CONST
    | nested_list                                               #NESTED
    | arith_atom                                                #ARITH
    ;

nested_list
    : LPAREN constant (COMMA constant)+ RPAREN
    | NUMBER
    | LPAREN nested_list (COMMA nested_list)+ RPAREN
   ;

def_name
    : IDENTIFIER
    ;

conditional
    : bool_expr 
    ;

spec_conditions
    : (LBRACK spec_condition RBRACK)*  #SPEC_LEVEL
    ;

spec_condition
    : ('@roof' | '@all')        #SPEC_ROOF
    | NUMBER                    #SPEC_SINGLE
    | NUMBER COLON NUMBER       #SPEC_FULL_INDX
    | NUMBER COLON              #SPEC_LEFT_INDX
    | COLON NUMBER              #SPEC_RIGHT_INDX
    ;

condition
    : bool_expr 
    ;

bool_expr
    :   ari_lparen bool_expr ari_rparen
    |   notop bool_expr
    |   bool_expr logicop bool_expr
    |   cmp_expr
    |   in_expr
    |   arith_atom
    ;

cmp_expr
    : arith_expr relop arith_expr (relop arith_expr)*
    ;

in_expr
    : arith_expr inop nested_list #INNESTED
    ;

arith_expr
    : ari_lparen arith_expr ari_rparen
    | arith_expr arith_op arith_expr
    | arith_atom 
    ;

arith_atom
    : 'item' '.' IDENTIFIER ('.' IDENTIFIER)*               # ATOM_SINGLE
    | 'item' '.' IDENTIFIER                                 # ATOM_SINGLE
    | 'item' '.' IDENTIFIER LBRACK STRING_LITERAL RBRACK    # ATOM_FROMATTR
    | 'item' LBRACK STRING_LITERAL RBRACK                   # ATOM_FROMATTR_SHORT
    | 'style' '.' IDENTIFIER                                # ATOM_STYLE
    | identifier                                            # ATOM_IDENT
    | NUMBER                                                # ATOM_IDENT
    | STRING_LITERAL                                        # ATOM_IDENT
    ;

ari_lparen
    : LPAREN
    ;

ari_rparen
    : RPAREN
    ;

const_atom
    : STRING_LITERAL | NUMBER | IDENTIFIER 
    ;

constant
    : 'rgb' LPAREN NUMBER COMMA NUMBER COMMA NUMBER RPAREN 
    | 'rgba' LPAREN NUMBER COMMA NUMBER COMMA NUMBER COMMA NUMBER RPAREN
    | HEX_NUMBER 
    | STRING_LITERAL 
    | NUMBER 
    | IDENTIFIER
    ;

simple_expr
    : identifier
    | NUMBER
    | STRING_LITERAL
    ; 
 
element_name
    : IDENTIFIER
    ;      

attr_name
    : IDENTIFIER
    ;  

identifier      // hack for attributes "faces" and "sharp_edges"
    : IDENTIFIER  
    ;  

relop
    : GT | GE | LT | LE | EQ
    ;

logicop
    : AND | OR
    ;

notop
    : NOT
    ;

inop
    : IN
    ;


arith_op
    : PLUS | MINUS | TIMES | DIV
    ;

number
    : NUMBER
    ; 

string_literal
    : STRING_LITERAL
    ;

// Lexer rules
// -------------------------------------

// in front so that they get not eaten by IDENTIFIER !!??
OR          : 'or';
AND         : 'and';
NOT         : 'not';
IN          : 'in';

IDENTIFIER
    : [a-zA-Z_]([a-zA-Z0-9_]|'-')*
    ;

STRING_LITERAL
    : '"' ('""' | ~ ('"'))* '"'
    ;

HEX_NUMBER 
    : '#' ('0'..'9'|'a'..'f'|'A'..'F')+
    ;

NUMBER
    : '-'? INT
    | '-'? FLOAT
    ; 

FLOAT
    : INT '.' INT*
    ;

INT
    : ('0' .. '9')+
    ;

STRUDEL:    '@';
LCURLY:     '{';
RCURLY:     '}';
LPAREN:     '(' ;
RPAREN:     ')' ;
LBRACK:     '[';
RBRACK:     ']';
PIPE:       '|';
COMMA:      ',' ;
COLON:      ':'; 
SEMI:       ';'; 

PLUS        : '+';
MINUS       : '-';
TIMES       : '*';
DIV         : '/';

GT          : '>' ;
GE          : '>=' ;
LT          : '<' ;
LE          : '<=' ;
EQ          : '==' ;

COMMENT
   :'//' .*? [\r\n] -> skip 
   ;  

WS : [ \t\r\n]+ -> skip ;
