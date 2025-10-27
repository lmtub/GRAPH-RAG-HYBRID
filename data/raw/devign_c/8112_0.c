static void hds_free(AVFormatContext *s)

{

    HDSContext *c = s->priv_data;

    int i, j;

    if (!c->streams)

        return;

    for (i = 0; i < s->nb_streams; i++) {

        OutputStream *os = &c->streams[i];

        if (os->out)

            avio_close(os->out);

        os->out = NULL;

        if (os->ctx && os->ctx_inited)

            av_write_trailer(os->ctx);

        if (os->ctx && os->ctx->pb)

            av_free(os->ctx->pb);

        if (os->ctx)

            avformat_free_context(os->ctx);

        av_free(os->metadata);

        for (j = 0; j < os->nb_extra_packets; j++)

            av_free(os->extra_packets[j]);

        for (j = 0; j < os->nb_fragments; j++)

            av_free(os->fragments[j]);

        av_free(os->fragments);

    }

    av_freep(&c->streams);

}
