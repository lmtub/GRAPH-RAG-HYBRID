static void json_message_process_token(JSONLexer *lexer, GString *input,

                                       JSONTokenType type, int x, int y)

{

    JSONMessageParser *parser = container_of(lexer, JSONMessageParser, lexer);

    QDict *dict;



    switch (type) {

    case JSON_LCURLY:

        parser->brace_count++;

        break;

    case JSON_RCURLY:

        parser->brace_count--;

        break;

    case JSON_LSQUARE:

        parser->bracket_count++;

        break;

    case JSON_RSQUARE:

        parser->bracket_count--;

        break;

    default:

        break;

    }



    dict = qdict_new();

    qdict_put(dict, "type", qint_from_int(type));

    qdict_put(dict, "token", qstring_from_str(input->str));

    qdict_put(dict, "x", qint_from_int(x));

    qdict_put(dict, "y", qint_from_int(y));



    parser->token_size += input->len;



    g_queue_push_tail(parser->tokens, dict);



    if (type == JSON_ERROR) {

        goto out_emit_bad;

    } else if (parser->brace_count < 0 ||

        parser->bracket_count < 0 ||

        (parser->brace_count == 0 &&

         parser->bracket_count == 0)) {

        goto out_emit;

    } else if (parser->token_size > MAX_TOKEN_SIZE ||

               parser->bracket_count + parser->brace_count > MAX_NESTING) {

        /* Security consideration, we limit total memory allocated per object

         * and the maximum recursion depth that a message can force.

         */

        goto out_emit_bad;

    }



    return;



out_emit_bad:

    /*

     * Clear out token list and tell the parser to emit an error

     * indication by passing it a NULL list

     */

    json_message_free_tokens(parser);

out_emit:

    /* send current list of tokens to parser and reset tokenizer */

    parser->brace_count = 0;

    parser->bracket_count = 0;

    /* parser->emit takes ownership of parser->tokens.  */

    parser->emit(parser, parser->tokens);

    parser->tokens = g_queue_new();

    parser->token_size = 0;

}
