static int amr_wb_encode_frame(AVCodecContext *avctx, AVPacket *avpkt,

                               const AVFrame *frame, int *got_packet_ptr)

{

    AMRWBContext *s = avctx->priv_data;

    const int16_t *samples = (const int16_t *)frame->data[0];

    int size, ret;



    if ((ret = ff_alloc_packet2(avctx, avpkt, MAX_PACKET_SIZE)))

        return ret;



    if (s->last_bitrate != avctx->bit_rate) {

        s->mode         = get_wb_bitrate_mode(avctx->bit_rate, avctx);

        s->last_bitrate = avctx->bit_rate;

    }

    size = E_IF_encode(s->state, s->mode, samples, avpkt->data, s->allow_dtx);

    if (size <= 0 || size > MAX_PACKET_SIZE) {

        av_log(avctx, AV_LOG_ERROR, "Error encoding frame\n");

        return AVERROR(EINVAL);

    }



    if (frame->pts != AV_NOPTS_VALUE)

        avpkt->pts = frame->pts - ff_samples_to_time_base(avctx, avctx->delay);



    avpkt->size = size;

    *got_packet_ptr = 1;

    return 0;

}
