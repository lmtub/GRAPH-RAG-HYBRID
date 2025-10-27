int json_lexer_flush(JSONLexer *lexer)

{

    return lexer->state == IN_START ? 0 : json_lexer_feed_char(lexer, 0);

}
