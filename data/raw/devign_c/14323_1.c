static int videotoolbox_common_end_frame(AVCodecContext *avctx, AVFrame *frame)

{

    int status;

    AVVideotoolboxContext *videotoolbox = avctx->hwaccel_context;

    VTContext *vtctx = avctx->internal->hwaccel_priv_data;



    av_buffer_unref(&frame->buf[0]);



    if (!videotoolbox->session || !vtctx->bitstream)

        return AVERROR_INVALIDDATA;



    status = videotoolbox_session_decode_frame(avctx);



    if (status) {

        av_log(avctx, AV_LOG_ERROR, "Failed to decode frame (%d)\n", status);

        return AVERROR_UNKNOWN;

    }



    if (!vtctx->frame)

        return AVERROR_UNKNOWN;



    return ff_videotoolbox_buffer_create(vtctx, frame);

}
