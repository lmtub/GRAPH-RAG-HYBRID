static int libopenjpeg_encode_frame(AVCodecContext *avctx, AVPacket *pkt,

                                    const AVFrame *frame, int *got_packet)

{

    LibOpenJPEGContext *ctx = avctx->priv_data;

    opj_cinfo_t *compress = ctx->compress;

    opj_image_t *image    = ctx->image;

    opj_cio_t *stream     = ctx->stream;

    int cpyresult = 0;

    int ret, len;

    AVFrame *gbrframe;



    switch (avctx->pix_fmt) {

    case AV_PIX_FMT_RGB24:

    case AV_PIX_FMT_RGBA:

    case AV_PIX_FMT_GRAY8A:

        cpyresult = libopenjpeg_copy_packed8(avctx, frame, image);

        break;

    case AV_PIX_FMT_XYZ12:

        cpyresult = libopenjpeg_copy_packed12(avctx, frame, image);

        break;

    case AV_PIX_FMT_RGB48:

    case AV_PIX_FMT_RGBA64:

        cpyresult = libopenjpeg_copy_packed16(avctx, frame, image);

        break;

    case AV_PIX_FMT_GBR24P:

    case AV_PIX_FMT_GBRP9:

    case AV_PIX_FMT_GBRP10:

    case AV_PIX_FMT_GBRP12:

    case AV_PIX_FMT_GBRP14:

    case AV_PIX_FMT_GBRP16:

        gbrframe = av_frame_alloc();



        av_frame_ref(gbrframe, frame);

        gbrframe->data[0] = frame->data[2]; // swap to be rgb

        gbrframe->data[1] = frame->data[0];

        gbrframe->data[2] = frame->data[1];

        gbrframe->linesize[0] = frame->linesize[2];

        gbrframe->linesize[1] = frame->linesize[0];

        gbrframe->linesize[2] = frame->linesize[1];

        if (avctx->pix_fmt == AV_PIX_FMT_GBR24P) {

            cpyresult = libopenjpeg_copy_unpacked8(avctx, gbrframe, image);

        } else {

            cpyresult = libopenjpeg_copy_unpacked16(avctx, gbrframe, image);

        }

        av_frame_free(&gbrframe);

        break;

    case AV_PIX_FMT_GRAY8:

    case AV_PIX_FMT_YUV410P:

    case AV_PIX_FMT_YUV411P:

    case AV_PIX_FMT_YUV420P:

    case AV_PIX_FMT_YUV422P:

    case AV_PIX_FMT_YUV440P:

    case AV_PIX_FMT_YUV444P:

    case AV_PIX_FMT_YUVA420P:

    case AV_PIX_FMT_YUVA422P:

    case AV_PIX_FMT_YUVA444P:

        cpyresult = libopenjpeg_copy_unpacked8(avctx, frame, image);

        break;

    case AV_PIX_FMT_GRAY16:

    case AV_PIX_FMT_YUV420P9:

    case AV_PIX_FMT_YUV422P9:

    case AV_PIX_FMT_YUV444P9:

    case AV_PIX_FMT_YUVA420P9:

    case AV_PIX_FMT_YUVA422P9:

    case AV_PIX_FMT_YUVA444P9:

    case AV_PIX_FMT_YUV444P10:

    case AV_PIX_FMT_YUV422P10:

    case AV_PIX_FMT_YUV420P10:

    case AV_PIX_FMT_YUVA444P10:

    case AV_PIX_FMT_YUVA422P10:

    case AV_PIX_FMT_YUVA420P10:

    case AV_PIX_FMT_YUV420P12:

    case AV_PIX_FMT_YUV422P12:

    case AV_PIX_FMT_YUV444P12:

    case AV_PIX_FMT_YUV420P14:

    case AV_PIX_FMT_YUV422P14:

    case AV_PIX_FMT_YUV444P14:

    case AV_PIX_FMT_YUV444P16:

    case AV_PIX_FMT_YUV422P16:

    case AV_PIX_FMT_YUV420P16:

    case AV_PIX_FMT_YUVA444P16:

    case AV_PIX_FMT_YUVA422P16:

    case AV_PIX_FMT_YUVA420P16:

        cpyresult = libopenjpeg_copy_unpacked16(avctx, frame, image);

        break;

    default:

        av_log(avctx, AV_LOG_ERROR,

               "The frame's pixel format '%s' is not supported\n",

               av_get_pix_fmt_name(avctx->pix_fmt));

        return AVERROR(EINVAL);

        break;

    }



    if (!cpyresult) {

        av_log(avctx, AV_LOG_ERROR,

               "Could not copy the frame data to the internal image buffer\n");

        return -1;

    }



    cio_seek(stream, 0);

    if (!opj_encode(compress, stream, image, NULL)) {

        av_log(avctx, AV_LOG_ERROR, "Error during the opj encode\n");

        return -1;

    }



    len = cio_tell(stream);

    if ((ret = ff_alloc_packet2(avctx, pkt, len)) < 0) {

        return ret;

    }



    memcpy(pkt->data, stream->buffer, len);

    pkt->flags |= AV_PKT_FLAG_KEY;

    *got_packet = 1;

    return 0;

}