static void start_frame(AVFilterLink *inlink, AVFilterBufferRef *picref)

{

    AVFilterContext *ctx = inlink->dst;

    TInterlaceContext *tinterlace = ctx->priv;



    if (tinterlace->cur)

        avfilter_unref_buffer(tinterlace->cur);

    tinterlace->cur  = tinterlace->next;

    tinterlace->next = picref;

}
