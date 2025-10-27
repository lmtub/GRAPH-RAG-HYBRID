static int apng_read_close(AVFormatContext *s)

{

    APNGDemuxContext *ctx = s->priv_data;

    av_freep(&ctx->extra_data);

    ctx->extra_data_size = 0;

    return 0;

}
