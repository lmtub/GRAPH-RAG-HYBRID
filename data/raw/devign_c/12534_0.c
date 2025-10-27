static QObject *parse_object(JSONParserContext *ctxt, va_list *ap)

{

    QDict *dict = NULL;

    QObject *token, *peek;

    JSONParserContext saved_ctxt = parser_context_save(ctxt);



    token = parser_context_pop_token(ctxt);

    if (token == NULL) {

        goto out;

    }



    if (!token_is_operator(token, '{')) {

        goto out;

    }



    dict = qdict_new();



    peek = parser_context_peek_token(ctxt);

    if (peek == NULL) {

        parse_error(ctxt, NULL, "premature EOI");

        goto out;

    }



    if (!token_is_operator(peek, '}')) {

        if (parse_pair(ctxt, dict, ap) == -1) {

            goto out;

        }



        token = parser_context_pop_token(ctxt);

        if (token == NULL) {

            parse_error(ctxt, NULL, "premature EOI");

            goto out;

        }



        while (!token_is_operator(token, '}')) {

            if (!token_is_operator(token, ',')) {

                parse_error(ctxt, token, "expected separator in dict");

                goto out;

            }



            if (parse_pair(ctxt, dict, ap) == -1) {

                goto out;

            }



            token = parser_context_pop_token(ctxt);

            if (token == NULL) {

                parse_error(ctxt, NULL, "premature EOI");

                goto out;

            }

        }

    } else {

        (void)parser_context_pop_token(ctxt);

    }



    return QOBJECT(dict);



out:

    parser_context_restore(ctxt, saved_ctxt);

    QDECREF(dict);

    return NULL;

}
