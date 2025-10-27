int qsv_init(AVCodecContext *s)

{

    InputStream *ist = s->opaque;

    QSVContext  *qsv = ist->hwaccel_ctx;

    AVQSVContext *hwctx_dec;

    int ret;



    if (!qsv) {

        av_log(NULL, AV_LOG_ERROR, "QSV transcoding is not initialized. "

               "-hwaccel qsv should only be used for one-to-one QSV transcoding "

               "with no filters.\n");

        return AVERROR_BUG;

    }



    ret = init_opaque_surf(qsv);

    if (ret < 0)

        return ret;



    hwctx_dec = av_qsv_alloc_context();

    if (!hwctx_dec)

        return AVERROR(ENOMEM);



    hwctx_dec->session        = qsv->session;

    hwctx_dec->iopattern      = MFX_IOPATTERN_OUT_OPAQUE_MEMORY;

    hwctx_dec->ext_buffers    = qsv->ext_buffers;

    hwctx_dec->nb_ext_buffers = FF_ARRAY_ELEMS(qsv->ext_buffers);



    av_freep(&s->hwaccel_context);

    s->hwaccel_context = hwctx_dec;



    ist->hwaccel_get_buffer = qsv_get_buffer;

    ist->hwaccel_uninit     = qsv_uninit;



    return 0;

}
