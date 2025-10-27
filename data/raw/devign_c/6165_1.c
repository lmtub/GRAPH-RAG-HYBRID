static void uninit(AVFilterContext *ctx)

{

    ZScaleContext *s = ctx->priv;



    zimg_filter_graph_free(s->graph);


    av_freep(&s->tmp);

    s->tmp_size = 0;

}