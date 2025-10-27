static QObject *parse_value(JSONParserContext *ctxt, va_list *ap)

{

    QObject *token;



    token = parser_context_peek_token(ctxt);

    if (token == NULL) {

        parse_error(ctxt, NULL, "premature EOI");

        return NULL;

    }



    switch (token_get_type(token)) {

    case JSON_LCURLY:

        return parse_object(ctxt, ap);

    case JSON_LSQUARE:

        return parse_array(ctxt, ap);

    case JSON_ESCAPE:

        return parse_escape(ctxt, ap);

    case JSON_INTEGER:

    case JSON_FLOAT:

    case JSON_STRING:

        return parse_literal(ctxt);

    case JSON_KEYWORD:

        return parse_keyword(ctxt);

    default:

        parse_error(ctxt, token, "expecting value");

        return NULL;

    }

}
