int ff_qsv_enc_init(AVCodecContext *avctx, QSVEncContext *q)

{

    int opaque_alloc = 0;

    int ret;



    q->param.IOPattern  = MFX_IOPATTERN_IN_SYSTEM_MEMORY;

    q->param.AsyncDepth = q->async_depth;



    q->async_fifo = av_fifo_alloc((1 + q->async_depth) *

                                  (sizeof(AVPacket) + sizeof(mfxSyncPoint) + sizeof(mfxBitstream*)));

    if (!q->async_fifo)

        return AVERROR(ENOMEM);



    if (avctx->hwaccel_context) {

        AVQSVContext *qsv = avctx->hwaccel_context;



        q->session         = qsv->session;

        q->param.IOPattern = qsv->iopattern;



        opaque_alloc = qsv->opaque_alloc;

    }



    if (!q->session) {

        ret = ff_qsv_init_internal_session(avctx, &q->internal_session,

                                           q->load_plugins);

        if (ret < 0)

            return ret;



        q->session = q->internal_session;

    }



    ret = init_video_param(avctx, q);

    if (ret < 0)

        return ret;



    ret = MFXVideoENCODE_QueryIOSurf(q->session, &q->param, &q->req);

    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error querying the encoding parameters\n");

        return ff_qsv_error(ret);

    }



    if (opaque_alloc) {

        ret = qsv_init_opaque_alloc(avctx, q);

        if (ret < 0)

            return ret;

    }



    if (avctx->hwaccel_context) {

        AVQSVContext *qsv = avctx->hwaccel_context;

        int i, j;



        q->extparam = av_mallocz_array(qsv->nb_ext_buffers + q->nb_extparam_internal,

                                       sizeof(*q->extparam));

        if (!q->extparam)

            return AVERROR(ENOMEM);



        q->param.ExtParam = q->extparam;

        for (i = 0; i < qsv->nb_ext_buffers; i++)

            q->param.ExtParam[i] = qsv->ext_buffers[i];

        q->param.NumExtParam = qsv->nb_ext_buffers;



        for (i = 0; i < q->nb_extparam_internal; i++) {

            for (j = 0; j < qsv->nb_ext_buffers; j++) {

                if (qsv->ext_buffers[j]->BufferId == q->extparam_internal[i]->BufferId)

                    break;

            }

            if (j < qsv->nb_ext_buffers)

                continue;



            q->param.ExtParam[q->param.NumExtParam++] = q->extparam_internal[i];

        }

    } else {

        q->param.ExtParam    = q->extparam_internal;

        q->param.NumExtParam = q->nb_extparam_internal;

    }



    ret = MFXVideoENCODE_Init(q->session, &q->param);

    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error initializing the encoder\n");

        return ff_qsv_error(ret);

    }



    ret = qsv_retrieve_enc_params(avctx, q);

    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error retrieving encoding parameters.\n");

        return ret;

    }



    q->avctx = avctx;



    return 0;

}
