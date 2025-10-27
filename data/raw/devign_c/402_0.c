void json_lexer_init(JSONLexer *lexer, JSONLexerEmitter func)

{

    lexer->emit = func;

    lexer->state = IN_START;

    lexer->token = qstring_new();

    lexer->x = lexer->y = 0;

}
