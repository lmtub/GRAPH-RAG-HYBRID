static int libquvi_read_header(AVFormatContext *s)

{

    int i, ret;

    quvi_t q;

    quvi_media_t m;

    QUVIcode rc;

    LibQuviContext *qc = s->priv_data;

    char *media_url, *pagetitle;



    rc = quvi_init(&q);

    if (rc != QUVI_OK)

        goto quvi_fail;



    quvi_setopt(q, QUVIOPT_FORMAT, qc->format);



    rc = quvi_parse(q, s->filename, &m);

    if (rc != QUVI_OK)

        goto quvi_fail;



    rc = quvi_getprop(m, QUVIPROP_MEDIAURL, &media_url);

    if (rc != QUVI_OK)

        goto quvi_fail;



    av_assert0(!qc->fmtctx->codec_whitelist && !qc->fmtctx->format_whitelist);

    qc->fmtctx-> codec_whitelist = av_strdup(s->codec_whitelist);

    qc->fmtctx->format_whitelist = av_strdup(s->format_whitelist);



    ret = avformat_open_input(&qc->fmtctx, media_url, NULL, NULL);

    if (ret < 0)

        goto end;



    rc = quvi_getprop(m, QUVIPROP_PAGETITLE, &pagetitle);

    if (rc == QUVI_OK)

        av_dict_set(&s->metadata, "title", pagetitle, 0);



    for (i = 0; i < qc->fmtctx->nb_streams; i++) {

        AVStream *st = avformat_new_stream(s, NULL);

        AVStream *ist = qc->fmtctx->streams[i];

        if (!st) {

            ret = AVERROR(ENOMEM);

            goto end;

        }

        avpriv_set_pts_info(st, ist->pts_wrap_bits, ist->time_base.num, ist->time_base.den);

        avcodec_copy_context(st->codec, qc->fmtctx->streams[i]->codec);

    }



    return 0;



quvi_fail:

    av_log(s, AV_LOG_ERROR, "%s\n", quvi_strerror(q, rc));

    ret = AVERROR_EXTERNAL;



end:

    quvi_parse_close(&m);

    quvi_close(&q);

    return ret;

}
