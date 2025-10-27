static int parse_pair(JSONParserContext *ctxt, QDict *dict, va_list *ap)

{

    QObject *key = NULL, *token = NULL, *value, *peek;

    JSONParserContext saved_ctxt = parser_context_save(ctxt);



    peek = parser_context_peek_token(ctxt);

    if (peek == NULL) {

        parse_error(ctxt, NULL, "premature EOI");

        goto out;

    }



    key = parse_value(ctxt, ap);

    if (!key || qobject_type(key) != QTYPE_QSTRING) {

        parse_error(ctxt, peek, "key is not a string in object");

        goto out;

    }



    token = parser_context_pop_token(ctxt);

    if (token == NULL) {

        parse_error(ctxt, NULL, "premature EOI");

        goto out;

    }



    if (!token_is_operator(token, ':')) {

        parse_error(ctxt, token, "missing : in object pair");

        goto out;

    }



    value = parse_value(ctxt, ap);

    if (value == NULL) {

        parse_error(ctxt, token, "Missing value in dict");

        goto out;

    }



    qdict_put_obj(dict, qstring_get_str(qobject_to_qstring(key)), value);



    qobject_decref(key);



    return 0;



out:

    parser_context_restore(ctxt, saved_ctxt);

    qobject_decref(key);



    return -1;

}
