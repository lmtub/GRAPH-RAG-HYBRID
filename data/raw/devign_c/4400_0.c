static int parse_pair(JSONParserContext *ctxt, QDict *dict, QList **tokens, va_list *ap)

{

    QObject *key = NULL, *token = NULL, *value, *peek;

    QList *working = qlist_copy(*tokens);



    peek = qlist_peek(working);

    if (peek == NULL) {

        parse_error(ctxt, NULL, "premature EOI");

        goto out;

    }



    key = parse_value(ctxt, &working, ap);

    if (!key || qobject_type(key) != QTYPE_QSTRING) {

        parse_error(ctxt, peek, "key is not a string in object");

        goto out;

    }



    token = qlist_pop(working);

    if (token == NULL) {

        parse_error(ctxt, NULL, "premature EOI");

        goto out;

    }



    if (!token_is_operator(token, ':')) {

        parse_error(ctxt, token, "missing : in object pair");

        goto out;

    }



    value = parse_value(ctxt, &working, ap);

    if (value == NULL) {

        parse_error(ctxt, token, "Missing value in dict");

        goto out;

    }



    qdict_put_obj(dict, qstring_get_str(qobject_to_qstring(key)), value);



    qobject_decref(token);

    qobject_decref(key);

    QDECREF(*tokens);

    *tokens = working;



    return 0;



out:

    qobject_decref(token);

    qobject_decref(key);

    QDECREF(working);



    return -1;

}
