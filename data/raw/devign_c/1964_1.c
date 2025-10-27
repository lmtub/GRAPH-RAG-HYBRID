static void dash_free(AVFormatContext *s)
{
    DASHContext *c = s->priv_data;
    int i, j;
    if (c->as) {
        for (i = 0; i < c->nb_as; i++)
            av_dict_free(&c->as[i].metadata);
        av_freep(&c->as);
        c->nb_as = 0;
    }
    if (!c->streams)
        return;
    for (i = 0; i < s->nb_streams; i++) {
        OutputStream *os = &c->streams[i];
        if (os->ctx && os->ctx_inited)
            av_write_trailer(os->ctx);
        if (os->ctx && os->ctx->pb)
            ffio_free_dyn_buf(&os->ctx->pb);
        ff_format_io_close(s, &os->out);
        if (os->ctx)
            avformat_free_context(os->ctx);
        for (j = 0; j < os->nb_segments; j++)
            av_free(os->segments[j]);
        av_free(os->segments);
    }
    av_freep(&c->streams);
}