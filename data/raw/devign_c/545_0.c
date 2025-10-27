static QObject *parse_array(JSONParserContext *ctxt, va_list *ap)

{

    QList *list = NULL;

    QObject *token, *peek;



    token = parser_context_pop_token(ctxt);

    assert(token && token_get_type(token) == JSON_LSQUARE);



    list = qlist_new();



    peek = parser_context_peek_token(ctxt);

    if (peek == NULL) {

        parse_error(ctxt, NULL, "premature EOI");

        goto out;

    }



    if (token_get_type(peek) != JSON_RSQUARE) {

        QObject *obj;



        obj = parse_value(ctxt, ap);

        if (obj == NULL) {

            parse_error(ctxt, token, "expecting value");

            goto out;

        }



        qlist_append_obj(list, obj);



        token = parser_context_pop_token(ctxt);

        if (token == NULL) {

            parse_error(ctxt, NULL, "premature EOI");

            goto out;

        }



        while (token_get_type(token) != JSON_RSQUARE) {

            if (token_get_type(token) != JSON_COMMA) {

                parse_error(ctxt, token, "expected separator in list");

                goto out;

            }



            obj = parse_value(ctxt, ap);

            if (obj == NULL) {

                parse_error(ctxt, token, "expecting value");

                goto out;

            }



            qlist_append_obj(list, obj);



            token = parser_context_pop_token(ctxt);

            if (token == NULL) {

                parse_error(ctxt, NULL, "premature EOI");

                goto out;

            }

        }

    } else {

        (void)parser_context_pop_token(ctxt);

    }



    return QOBJECT(list);



out:

    QDECREF(list);

    return NULL;

}
