QObject *json_parser_parse_err(QList *tokens, va_list *ap, Error **errp)

{

    JSONParserContext ctxt = {};

    QList *working = qlist_copy(tokens);

    QObject *result;



    result = parse_value(&ctxt, &working, ap);



    QDECREF(working);



    error_propagate(errp, ctxt.err);



    return result;

}
