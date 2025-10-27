static int libopus_encode(AVCodecContext *avctx, AVPacket *avpkt,

                          const AVFrame *frame, int *got_packet_ptr)

{

    LibopusEncContext *opus = avctx->priv_data;

    const int sample_size   = avctx->channels *

                              av_get_bytes_per_sample(avctx->sample_fmt);

    uint8_t *audio;

    int ret;

    int discard_padding;



    if (frame) {

        ret = ff_af_queue_add(&opus->afq, frame);

        if (ret < 0)

            return ret;

        if (frame->nb_samples < opus->opts.packet_size) {

            audio = opus->samples;

            memcpy(audio, frame->data[0], frame->nb_samples * sample_size);

        } else

            audio = frame->data[0];

    } else {

        if (!opus->afq.remaining_samples)

            return 0;

        audio = opus->samples;

        memset(audio, 0, opus->opts.packet_size * sample_size);

    }



    /* Maximum packet size taken from opusenc in opus-tools. 60ms packets

     * consist of 3 frames in one packet. The maximum frame size is 1275

     * bytes along with the largest possible packet header of 7 bytes. */

    if ((ret = ff_alloc_packet2(avctx, avpkt, (1275 * 3 + 7) * opus->stream_count, 0)) < 0)

        return ret;



    if (avctx->sample_fmt == AV_SAMPLE_FMT_FLT)

        ret = opus_multistream_encode_float(opus->enc, (float *)audio,

                                            opus->opts.packet_size,

                                            avpkt->data, avpkt->size);

    else

        ret = opus_multistream_encode(opus->enc, (opus_int16 *)audio,

                                      opus->opts.packet_size,

                                      avpkt->data, avpkt->size);



    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR,

               "Error encoding frame: %s\n", opus_strerror(ret));

        return ff_opus_error_to_averror(ret);

    }



    av_shrink_packet(avpkt, ret);



    ff_af_queue_remove(&opus->afq, opus->opts.packet_size,

                       &avpkt->pts, &avpkt->duration);



    discard_padding = opus->opts.packet_size - avpkt->duration;

    // Check if subtraction resulted in an overflow

    if ((discard_padding < opus->opts.packet_size) != (avpkt->duration > 0)) {

        av_free_packet(avpkt);

        av_free(avpkt);

        return AVERROR(EINVAL);

    }

    if (discard_padding > 0) {

        uint8_t* side_data = av_packet_new_side_data(avpkt,

                                                     AV_PKT_DATA_SKIP_SAMPLES,

                                                     10);

        if(!side_data) {

            av_free_packet(avpkt);

            av_free(avpkt);

            return AVERROR(ENOMEM);

        }

        AV_WL32(side_data + 4, discard_padding);

    }



    *got_packet_ptr = 1;



    return 0;

}
