static QObject *parser_context_pop_token(JSONParserContext *ctxt)

{

    qobject_decref(ctxt->current);

    assert(!g_queue_is_empty(ctxt->buf));

    ctxt->current = g_queue_pop_head(ctxt->buf);

    return ctxt->current;

}
