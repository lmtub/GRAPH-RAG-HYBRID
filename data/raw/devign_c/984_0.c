static void parse_error(JSONParserContext *ctxt, QObject *token, const char *msg, ...)

{

    fprintf(stderr, "parse error: %s\n", msg);

}
