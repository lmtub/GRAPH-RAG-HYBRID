static QObject *parse_keyword(JSONParserContext *ctxt)

{

    QObject *token, *ret;

    JSONParserContext saved_ctxt = parser_context_save(ctxt);



    token = parser_context_pop_token(ctxt);

    if (token == NULL) {

        goto out;

    }



    if (token_get_type(token) != JSON_KEYWORD) {

        goto out;

    }



    if (token_is_keyword(token, "true")) {

        ret = QOBJECT(qbool_from_int(true));

    } else if (token_is_keyword(token, "false")) {

        ret = QOBJECT(qbool_from_int(false));



    } else {

        parse_error(ctxt, token, "invalid keyword `%s'", token_get_value(token));

        goto out;

    }



    return ret;



out: 

    parser_context_restore(ctxt, saved_ctxt);



    return NULL;

}