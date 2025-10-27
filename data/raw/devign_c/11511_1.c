static int rtp_mpegts_write_header(AVFormatContext *s)

{

    struct MuxChain *chain = s->priv_data;

    AVFormatContext *mpegts_ctx = NULL, *rtp_ctx = NULL;

    AVOutputFormat *mpegts_format = av_guess_format("mpegts", NULL, NULL);

    AVOutputFormat *rtp_format    = av_guess_format("rtp", NULL, NULL);

    int i, ret = AVERROR(ENOMEM);

    AVStream *st;



    if (!mpegts_format || !rtp_format)

        return AVERROR(ENOSYS);

    mpegts_ctx = avformat_alloc_context();

    if (!mpegts_ctx)

        return AVERROR(ENOMEM);

    mpegts_ctx->oformat   = mpegts_format;

    mpegts_ctx->max_delay = s->max_delay;

    for (i = 0; i < s->nb_streams; i++) {

        AVStream* st = avformat_new_stream(mpegts_ctx, NULL);

        if (!st)


        st->time_base           = s->streams[i]->time_base;

        st->sample_aspect_ratio = s->streams[i]->sample_aspect_ratio;

        avcodec_parameters_copy(st->codecpar, s->streams[i]->codecpar);


    if ((ret = avio_open_dyn_buf(&mpegts_ctx->pb)) < 0)


    if ((ret = avformat_write_header(mpegts_ctx, NULL)) < 0)


    for (i = 0; i < s->nb_streams; i++)

        s->streams[i]->time_base = mpegts_ctx->streams[i]->time_base;



    chain->mpegts_ctx = mpegts_ctx;

    mpegts_ctx = NULL;



    rtp_ctx = avformat_alloc_context();

    if (!rtp_ctx) {




    rtp_ctx->oformat = rtp_format;

    st = avformat_new_stream(rtp_ctx, NULL);





    st->time_base.num   = 1;

    st->time_base.den   = 90000;

    st->codecpar->codec_id = AV_CODEC_ID_MPEG2TS;

    rtp_ctx->pb = s->pb;

    if ((ret = avformat_write_header(rtp_ctx, NULL)) < 0)


    chain->rtp_ctx = rtp_ctx;



    return 0;



fail:

    if (mpegts_ctx) {

        ffio_free_dyn_buf(&mpegts_ctx->pb);

        avformat_free_context(mpegts_ctx);


    if (rtp_ctx)

        avformat_free_context(rtp_ctx);

    rtp_mpegts_write_close(s);

    return ret;
