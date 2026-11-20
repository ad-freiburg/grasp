%start term

%%

// only the lexer is used, to find terms in text
term
    : 'PREFIX'
    | 'IRIREF'
    | 'PNAME_NS'
    | 'PNAME_LN'
    ;
