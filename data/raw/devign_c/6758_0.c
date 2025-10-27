QObject *json_parser_parse_err(QList *tokens, va_list *ap, Error **errp)

{

    JSONParserContext ctxt = {};

    QList *working;

    QObject *result;



    if (!tokens) {

        return NULL;

    }

    working = qlist_copy(tokens);

    result = parse_value(&ctxt, &working, ap);



    QDECREF(working);



    error_propagate(errp, ctxt.err);



    return result;

}
