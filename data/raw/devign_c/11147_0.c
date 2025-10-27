static void GCC_FMT_ATTR(3, 4) parse_error(JSONParserContext *ctxt,

                                           QObject *token, const char *msg, ...)

{

    va_list ap;

    char message[1024];

    va_start(ap, msg);

    vsnprintf(message, sizeof(message), msg, ap);

    va_end(ap);

    if (ctxt->err) {

        error_free(ctxt->err);

        ctxt->err = NULL;

    }

    error_setg(&ctxt->err, "JSON parse error, %s", message);

}
