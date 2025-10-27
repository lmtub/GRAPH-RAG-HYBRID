static int qsv_decode_init(AVCodecContext *avctx, QSVContext *q, AVPacket *avpkt)

{

    mfxVideoParam param = { { 0 } };

    mfxBitstream bs   = { { { 0 } } };

    int ret;

    enum AVPixelFormat pix_fmts[3] = { AV_PIX_FMT_QSV,

                                       AV_PIX_FMT_NV12,

                                       AV_PIX_FMT_NONE };



    ret = ff_get_format(avctx, pix_fmts);

    if (ret < 0)

        return ret;



    avctx->pix_fmt      = ret;



    q->iopattern  = MFX_IOPATTERN_OUT_SYSTEM_MEMORY;

    if (avctx->hwaccel_context) {

        AVQSVContext *qsv = avctx->hwaccel_context;



        q->session        = qsv->session;

        q->iopattern      = qsv->iopattern;

        q->ext_buffers    = qsv->ext_buffers;

        q->nb_ext_buffers = qsv->nb_ext_buffers;

    }

    if (!q->session) {

        if (!q->internal_qs.session) {

            ret = ff_qsv_init_internal_session(avctx, &q->internal_qs,

                                               q->load_plugins);

            if (ret < 0)

                return ret;

        }



        q->session = q->internal_qs.session;

    }



    if (avpkt->size) {

        bs.Data       = avpkt->data;

        bs.DataLength = avpkt->size;

        bs.MaxLength  = bs.DataLength;

        bs.TimeStamp  = avpkt->pts;

    } else

        return AVERROR_INVALIDDATA;



    ret = ff_qsv_codec_id_to_mfx(avctx->codec_id);

    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "Unsupported codec_id %08x\n", avctx->codec_id);

        return ret;

    }



    param.mfx.CodecId = ret;



    ret = MFXVideoDECODE_DecodeHeader(q->session, &bs, &param);

    if (MFX_ERR_MORE_DATA==ret) {

        /* this code means that header not found so we return packet size to skip

           a current packet

         */

        return avpkt->size;

    } else if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "Decode header error %d\n", ret);

        return ff_qsv_error(ret);

    }

    param.IOPattern   = q->iopattern;

    param.AsyncDepth  = q->async_depth;

    param.ExtParam    = q->ext_buffers;

    param.NumExtParam = q->nb_ext_buffers;

    param.mfx.FrameInfo.BitDepthLuma   = 8;

    param.mfx.FrameInfo.BitDepthChroma = 8;



    ret = MFXVideoDECODE_Init(q->session, &param);

    if (ret < 0) {

        if (MFX_ERR_INVALID_VIDEO_PARAM==ret) {

            av_log(avctx, AV_LOG_ERROR,

                   "Error initializing the MFX video decoder, unsupported video\n");

        } else {

            av_log(avctx, AV_LOG_ERROR,

                   "Error initializing the MFX video decoder %d\n", ret);

        }

        return ff_qsv_error(ret);

    }



    avctx->profile      = param.mfx.CodecProfile;

    avctx->level        = param.mfx.CodecLevel;

    avctx->coded_width  = param.mfx.FrameInfo.Width;

    avctx->coded_height = param.mfx.FrameInfo.Height;

    avctx->width        = param.mfx.FrameInfo.CropW - param.mfx.FrameInfo.CropX;

    avctx->height       = param.mfx.FrameInfo.CropH - param.mfx.FrameInfo.CropY;



    /* maximum decoder latency should be not exceed max DPB size for h.264 and

       HEVC which is 16 for both cases.

       So weare  pre-allocating fifo big enough for 17 elements:

     */

    if (!q->async_fifo) {

        q->async_fifo = av_fifo_alloc((1 + 16) *

                                      (sizeof(mfxSyncPoint*) + sizeof(QSVFrame*)));

        if (!q->async_fifo)

            return AVERROR(ENOMEM);

    }



    if (!q->input_fifo) {

        q->input_fifo = av_fifo_alloc(1024*16);

        if (!q->input_fifo)

            return AVERROR(ENOMEM);

    }



    if (!q->pkt_fifo) {

        q->pkt_fifo = av_fifo_alloc( sizeof(AVPacket) * (1 + 16) );

        if (!q->pkt_fifo)

            return AVERROR(ENOMEM);

    }

    q->engine_ready = 1;



    return 0;

}
