static int do_decode(AVCodecContext *avctx, AVPacket *pkt)

{

    int got_frame;

    int ret;



    av_assert0(!avctx->internal->buffer_frame->buf[0]);



    if (!pkt)

        pkt = avctx->internal->buffer_pkt;



    // This is the lesser evil. The field is for compatibility with legacy users

    // of the legacy API, and users using the new API should not be forced to

    // even know about this field.

    avctx->refcounted_frames = 1;



    // Some codecs (at least wma lossless) will crash when feeding drain packets

    // after EOF was signaled.

    if (avctx->internal->draining_done)

        return AVERROR_EOF;



    if (avctx->codec_type == AVMEDIA_TYPE_VIDEO) {

        ret = avcodec_decode_video2(avctx, avctx->internal->buffer_frame,

                                    &got_frame, pkt);

        if (ret >= 0 && !(avctx->flags & AV_CODEC_FLAG_TRUNCATED))

            ret = pkt->size;

    } else if (avctx->codec_type == AVMEDIA_TYPE_AUDIO) {

        ret = avcodec_decode_audio4(avctx, avctx->internal->buffer_frame,

                                    &got_frame, pkt);

    } else {

        ret = AVERROR(EINVAL);

    }



    if (ret == AVERROR(EAGAIN))

        ret = pkt->size;



    if (ret < 0)

        return ret;



    if (avctx->internal->draining && !got_frame)

        avctx->internal->draining_done = 1;



    if (ret >= pkt->size) {

        av_packet_unref(avctx->internal->buffer_pkt);

    } else {

        int consumed = ret;



        if (pkt != avctx->internal->buffer_pkt) {

            av_packet_unref(avctx->internal->buffer_pkt);

            if ((ret = av_packet_ref(avctx->internal->buffer_pkt, pkt)) < 0)

                return ret;

        }



        avctx->internal->buffer_pkt->data += consumed;

        avctx->internal->buffer_pkt->size -= consumed;

        avctx->internal->buffer_pkt->pts   = AV_NOPTS_VALUE;

        avctx->internal->buffer_pkt->dts   = AV_NOPTS_VALUE;

    }



    if (got_frame)

        av_assert0(avctx->internal->buffer_frame->buf[0]);



    return 0;

}
