static QObject *parse_escape(JSONParserContext *ctxt, va_list *ap)

{

    QObject *token;

    const char *val;



    if (ap == NULL) {

        return NULL;

    }



    token = parser_context_pop_token(ctxt);

    assert(token && token_get_type(token) == JSON_ESCAPE);

    val = token_get_value(token);



    if (!strcmp(val, "%p")) {

        return va_arg(*ap, QObject *);

    } else if (!strcmp(val, "%i")) {

        return QOBJECT(qbool_from_bool(va_arg(*ap, int)));

    } else if (!strcmp(val, "%d")) {

        return QOBJECT(qint_from_int(va_arg(*ap, int)));

    } else if (!strcmp(val, "%ld")) {

        return QOBJECT(qint_from_int(va_arg(*ap, long)));

    } else if (!strcmp(val, "%lld") ||

               !strcmp(val, "%I64d")) {

        return QOBJECT(qint_from_int(va_arg(*ap, long long)));

    } else if (!strcmp(val, "%s")) {

        return QOBJECT(qstring_from_str(va_arg(*ap, const char *)));

    } else if (!strcmp(val, "%f")) {

        return QOBJECT(qfloat_from_double(va_arg(*ap, double)));

    }

    return NULL;

}
