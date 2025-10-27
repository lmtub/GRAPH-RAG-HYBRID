int ff_qsv_decode_init(AVCodecContext *avctx, QSVContext *q, mfxSession session)

{

    mfxVideoParam param = { { 0 } };

    int ret;



    q->async_fifo = av_fifo_alloc((1 + q->async_depth) *

                                  (sizeof(mfxSyncPoint) + sizeof(QSVFrame*)));

    if (!q->async_fifo)

        return AVERROR(ENOMEM);



    ret = qsv_init_session(avctx, q, session);

    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error initializing an MFX session\n");

        return ret;

    }





    ret = ff_qsv_codec_id_to_mfx(avctx->codec_id);

    if (ret < 0)

        return ret;



    param.mfx.CodecId      = ret;

    param.mfx.CodecProfile = avctx->profile;

    param.mfx.CodecLevel   = avctx->level;



    param.mfx.FrameInfo.BitDepthLuma   = 8;

    param.mfx.FrameInfo.BitDepthChroma = 8;

    param.mfx.FrameInfo.Shift          = 0;

    param.mfx.FrameInfo.FourCC         = MFX_FOURCC_NV12;

    param.mfx.FrameInfo.Width          = avctx->coded_width;

    param.mfx.FrameInfo.Height         = avctx->coded_height;

    param.mfx.FrameInfo.ChromaFormat   = MFX_CHROMAFORMAT_YUV420;



    param.IOPattern   = q->iopattern;

    param.AsyncDepth  = q->async_depth;

    param.ExtParam    = q->ext_buffers;

    param.NumExtParam = q->nb_ext_buffers;



    ret = MFXVideoDECODE_Init(q->session, &param);

    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error initializing the MFX video decoder\n");

        return ff_qsv_error(ret);

    }



    return 0;

}
