static int twolame_encode_frame(AVCodecContext *avctx, AVPacket *avpkt,

                                const AVFrame *frame, int *got_packet_ptr)

{

    TWOLAMEContext *s = avctx->priv_data;

    int ret;



    if ((ret = ff_alloc_packet(avpkt, MPA_MAX_CODED_FRAME_SIZE)) < 0)

        return ret;



    if (frame) {

        switch (avctx->sample_fmt) {

        case AV_SAMPLE_FMT_FLT:

            ret = twolame_encode_buffer_float32_interleaved(s->glopts,

                                                            (const float *)frame->data[0],

                                                            frame->nb_samples,

                                                            avpkt->data,

                                                            avpkt->size);

            break;

        case AV_SAMPLE_FMT_FLTP:

            ret = twolame_encode_buffer_float32(s->glopts,

                                                (const float *)frame->data[0],

                                                (const float *)frame->data[1],

                                                frame->nb_samples,

                                                avpkt->data, avpkt->size);

            break;

        case AV_SAMPLE_FMT_S16:

            ret = twolame_encode_buffer_interleaved(s->glopts,

                                                    (const short int *)frame->data[0],

                                                    frame->nb_samples,

                                                    avpkt->data, avpkt->size);

            break;

        case AV_SAMPLE_FMT_S16P:

            ret = twolame_encode_buffer(s->glopts,

                                        (const short int *)frame->data[0],

                                        (const short int *)frame->data[1],

                                        frame->nb_samples,

                                        avpkt->data, avpkt->size);

            break;

        default:

            av_log(avctx, AV_LOG_ERROR,

                   "Unsupported sample format %d.\n", avctx->sample_fmt);

            return AVERROR_BUG;

        }

    } else {

        ret = twolame_encode_flush(s->glopts, avpkt->data, avpkt->size);

    }



    if (!ret)     // no bytes written

        return 0;

    if (ret < 0)  // twolame error

        return AVERROR_UNKNOWN;



    avpkt->duration = ff_samples_to_time_base(avctx, frame->nb_samples);

    if (frame) {

        if (frame->pts != AV_NOPTS_VALUE)

            avpkt->pts = frame->pts - ff_samples_to_time_base(avctx, avctx->initial_padding);

    } else {

        avpkt->pts = s->next_pts;

    }

    // this is for setting pts for flushed packet(s).

    if (avpkt->pts != AV_NOPTS_VALUE)

        s->next_pts = avpkt->pts + avpkt->duration;



    av_shrink_packet(avpkt, ret);

    *got_packet_ptr = 1;

    return 0;

}
