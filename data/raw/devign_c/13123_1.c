int attribute_align_arg avcodec_encode_audio(AVCodecContext *avctx,

                                             uint8_t *buf, int buf_size,

                                             const short *samples)

{

    AVPacket pkt;

    AVFrame *frame;

    int ret, samples_size, got_packet;



    av_init_packet(&pkt);

    pkt.data = buf;

    pkt.size = buf_size;



    if (samples) {

        frame = av_frame_alloc();





        if (avctx->frame_size) {

            frame->nb_samples = avctx->frame_size;

        } else {

            /* if frame_size is not set, the number of samples must be

             * calculated from the buffer size */

            int64_t nb_samples;

            if (!av_get_bits_per_sample(avctx->codec_id)) {

                av_log(avctx, AV_LOG_ERROR, "avcodec_encode_audio() does not "

                                            "support this codec\n");

                av_frame_free(&frame);

                return AVERROR(EINVAL);

            }

            nb_samples = (int64_t)buf_size * 8 /

                         (av_get_bits_per_sample(avctx->codec_id) *

                          avctx->channels);

            if (nb_samples >= INT_MAX) {

                av_frame_free(&frame);

                return AVERROR(EINVAL);

            }

            frame->nb_samples = nb_samples;

        }



        /* it is assumed that the samples buffer is large enough based on the

         * relevant parameters */

        samples_size = av_samples_get_buffer_size(NULL, avctx->channels,

                                                  frame->nb_samples,

                                                  avctx->sample_fmt, 1);

        if ((ret = avcodec_fill_audio_frame(frame, avctx->channels,

                                            avctx->sample_fmt,

                                            (const uint8_t *)samples,

                                            samples_size, 1)) < 0) {

            av_frame_free(&frame);

            return ret;

        }



        /* fabricate frame pts from sample count.

         * this is needed because the avcodec_encode_audio() API does not have

         * a way for the user to provide pts */

        if (avctx->sample_rate && avctx->time_base.num)

            frame->pts = ff_samples_to_time_base(avctx,

                                                 avctx->internal->sample_count);

        else

            frame->pts = AV_NOPTS_VALUE;

        avctx->internal->sample_count += frame->nb_samples;

    } else {

        frame = NULL;

    }



    got_packet = 0;

    ret = avcodec_encode_audio2(avctx, &pkt, frame, &got_packet);

    if (!ret && got_packet && avctx->coded_frame) {

        avctx->coded_frame->pts       = pkt.pts;

        avctx->coded_frame->key_frame = !!(pkt.flags & AV_PKT_FLAG_KEY);

    }

    /* free any side data since we cannot return it */

    av_packet_free_side_data(&pkt);



    if (frame && frame->extended_data != frame->data)

        av_freep(&frame->extended_data);



    av_frame_free(&frame);

    return ret ? ret : pkt.size;

}