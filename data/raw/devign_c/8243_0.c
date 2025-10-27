static QObject *parser_context_peek_token(JSONParserContext *ctxt)

{

    assert(!g_queue_is_empty(ctxt->buf));

    return g_queue_peek_head(ctxt->buf);

}
