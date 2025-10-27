static QObject *parse_value(JSONParserContext *ctxt, QList **tokens, va_list *ap)

{

    QObject *obj;



    obj = parse_object(ctxt, tokens, ap);

    if (obj == NULL) {

        obj = parse_array(ctxt, tokens, ap);

    }

    if (obj == NULL) {

        obj = parse_escape(ctxt, tokens, ap);

    }

    if (obj == NULL) {

        obj = parse_keyword(ctxt, tokens);

    } 

    if (obj == NULL) {

        obj = parse_literal(ctxt, tokens);

    }



    return obj;

}
