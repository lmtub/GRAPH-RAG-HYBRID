static int start_frame(AVFilterLink *inlink, AVFilterBufferRef *picref)

{

    AVFilterContext *ctx = inlink->dst;

    TInterlaceContext *tinterlace = ctx->priv;



    avfilter_unref_buffer(tinterlace->cur);

    tinterlace->cur  = tinterlace->next;

    tinterlace->next = picref;


    return 0;

}