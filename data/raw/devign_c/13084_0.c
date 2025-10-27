int qsv_transcode_init(OutputStream *ost)

{

    InputStream *ist;

    const enum AVPixelFormat *pix_fmt;



    AVDictionaryEntry *e;

    const AVOption *opt;

    int flags = 0;



    int err, i;



    QSVContext *qsv = NULL;

    AVQSVContext *hwctx = NULL;

    mfxIMPL impl;

    mfxVersion ver = { { 3, 1 } };



    /* check if the encoder supports QSV */

    if (!ost->enc->pix_fmts)

        return 0;

    for (pix_fmt = ost->enc->pix_fmts; *pix_fmt != AV_PIX_FMT_NONE; pix_fmt++)

        if (*pix_fmt == AV_PIX_FMT_QSV)

            break;

    if (*pix_fmt == AV_PIX_FMT_NONE)

        return 0;



    if (strcmp(ost->avfilter, "null") || ost->source_index < 0)

        return 0;



    /* check if the decoder supports QSV and the output only goes to this stream */

    ist = input_streams[ost->source_index];

    if (ist->hwaccel_id != HWACCEL_QSV || !ist->dec || !ist->dec->pix_fmts)

        return 0;

    for (pix_fmt = ist->dec->pix_fmts; *pix_fmt != AV_PIX_FMT_NONE; pix_fmt++)

        if (*pix_fmt == AV_PIX_FMT_QSV)

            break;

    if (*pix_fmt == AV_PIX_FMT_NONE)

        return 0;



    for (i = 0; i < nb_output_streams; i++)

        if (output_streams[i] != ost &&

            output_streams[i]->source_index == ost->source_index)

            return 0;



    av_log(NULL, AV_LOG_VERBOSE, "Setting up QSV transcoding\n");



    qsv   = av_mallocz(sizeof(*qsv));

    hwctx = av_qsv_alloc_context();

    if (!qsv || !hwctx)

        goto fail;



    impl = choose_implementation(ist);



    err = MFXInit(impl, &ver, &qsv->session);

    if (err != MFX_ERR_NONE) {

        av_log(NULL, AV_LOG_ERROR, "Error initializing an MFX session: %d\n", err);

        goto fail;

    }



    e = av_dict_get(ost->encoder_opts, "flags", NULL, 0);

    opt = av_opt_find(ost->enc_ctx, "flags", NULL, 0, 0);

    if (e && opt)

        av_opt_eval_flags(ost->enc_ctx, opt, e->value, &flags);



    qsv->ost = ost;



    hwctx->session                = qsv->session;

    hwctx->iopattern              = MFX_IOPATTERN_IN_OPAQUE_MEMORY;

    hwctx->opaque_alloc           = 1;

    hwctx->nb_opaque_surfaces     = 16;



    ost->hwaccel_ctx              = qsv;

    ost->enc_ctx->hwaccel_context = hwctx;

    ost->enc_ctx->pix_fmt         = AV_PIX_FMT_QSV;



    ist->hwaccel_ctx              = qsv;

    ist->dec_ctx->pix_fmt         = AV_PIX_FMT_QSV;

    ist->resample_pix_fmt         = AV_PIX_FMT_QSV;



    return 0;



fail:

    av_freep(&hwctx);

    av_freep(&qsv);

    return AVERROR_UNKNOWN;

}
