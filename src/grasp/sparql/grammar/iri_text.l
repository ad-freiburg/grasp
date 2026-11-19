// IRIs, prefixed names and PREFIX keywords in text, everything else is
// skipped; tokens ported from https://www.w3.org/TR/sparql11-query/#sparqlGrammar

// fragments
WS \x20 | \x09 | \x0D | \x0A
PN_CHARS_BASE [A-Z] | [a-z] | [À-Ö] | [Ø-ö] | [ø-˿] | [Ͱ-ͽ] | [Ϳ-῿] | [‌-‍] | [⁰-↏] | [Ⰰ-⿯] | [、-퟿] | [豈-﷏] | [ﷰ-�] | [\U00010000-\U000EFFFF]
PN_CHARS_U {PN_CHARS_BASE} | '_'
PN_CHARS {PN_CHARS_U} | '-' | [0-9] | · | [̀-ͯ] | [‿-⁀]
PN_PREFIX {PN_CHARS_BASE} (( {PN_CHARS} | '.' )* {PN_CHARS})?
PN_LOCAL ({PN_CHARS_U} | ':' | [0-9] | {PLX} ) (({PN_CHARS} | '.' | ':' | {PLX})* ({PN_CHARS} | ':' | {PLX}) )?
PLX {PERCENT} | {PN_LOCAL_ESC}
PERCENT '%' {HEX} {HEX}
HEX [0-9] | [A-F] | [a-f]
PN_LOCAL_ESC '\' ( '_' | '~' | '.' | '-' | '!' | '$' | '&' | "'" | '(' | ')' | '*' | '+' | ',' | ';' | '=' | '/' | '?' | '#' | '@' | '%' )

%%

PREFIX [Pp] [Rr] [Ee] [Ff] [Ii] [Xx]
IRIREF '<' ([[^<>"{}|^`\\]--[\x00-\x20]])* '>'
PNAME_NS {PN_PREFIX}? ':'
PNAME_LN {PNAME_NS} {PN_LOCAL}

// whitespace and all other characters (ignored)
; {WS}+
; .
