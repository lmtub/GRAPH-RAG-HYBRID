static QObject *parse_literal(JSONParserContext *ctxt)

{

    QObject *token;



    token = parser_context_pop_token(ctxt);

    assert(token);



    switch (token_get_type(token)) {

    case JSON_STRING:

        return QOBJECT(qstring_from_escaped_str(ctxt, token));

    case JSON_INTEGER: {

        /* A possibility exists that this is a whole-valued float where the

         * fractional part was left out due to being 0 (.0). It's not a big

         * deal to treat these as ints in the parser, so long as users of the

         * resulting QObject know to expect a QInt in place of a QFloat in

         * cases like these.

         *

         * However, in some cases these values will overflow/underflow a

         * QInt/int64 container, thus we should assume these are to be handled

         * as QFloats/doubles rather than silently changing their values.

         *

         * strtoll() indicates these instances by setting errno to ERANGE

         */

        int64_t value;



        errno = 0; /* strtoll doesn't set errno on success */

        value = strtoll(token_get_value(token), NULL, 10);

        if (errno != ERANGE) {

            return QOBJECT(qint_from_int(value));

        }

        /* fall through to JSON_FLOAT */

    }

    case JSON_FLOAT:

        /* FIXME dependent on locale */

        return QOBJECT(qfloat_from_double(strtod(token_get_value(token),

                                                 NULL)));

    default:

        abort();

    }

}
