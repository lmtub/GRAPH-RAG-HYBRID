static void parser_context_free(JSONParserContext *ctxt)

{

    if (ctxt) {

        while (!g_queue_is_empty(ctxt->buf)) {

            parser_context_pop_token(ctxt);

        }

        qobject_decref(ctxt->current);

        g_queue_free(ctxt->buf);

        g_free(ctxt);

    }

}
