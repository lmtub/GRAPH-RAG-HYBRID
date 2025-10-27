static QObject *parse_keyword(JSONParserContext *ctxt, QList **tokens)

{

    QObject *token, *ret;

    QList *working = qlist_copy(*tokens);



    token = qlist_pop(working);

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



    qobject_decref(token);

    QDECREF(*tokens);

    *tokens = working;



    return ret;



out: 

    qobject_decref(token);

    QDECREF(working);



    return NULL;

}
