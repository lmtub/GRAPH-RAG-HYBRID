static int MPA_encode_frame(AVCodecContext *avctx, AVPacket *avpkt,

                            const AVFrame *frame, int *got_packet_ptr)

{

    MpegAudioContext *s = avctx->priv_data;

    const int16_t *samples = (const int16_t *)frame->data[0];

    short smr[MPA_MAX_CHANNELS][SBLIMIT];

    unsigned char bit_alloc[MPA_MAX_CHANNELS][SBLIMIT];

    int padding, i, ret;



    for(i=0;i<s->nb_channels;i++) {

        filter(s, i, samples + i, s->nb_channels);

    }



    for(i=0;i<s->nb_channels;i++) {

        compute_scale_factors(s, s->scale_code[i], s->scale_factors[i],

                              s->sb_samples[i], s->sblimit);

    }

    for(i=0;i<s->nb_channels;i++) {

        psycho_acoustic_model(s, smr[i]);

    }

    compute_bit_allocation(s, smr, bit_alloc, &padding);



    if ((ret = ff_alloc_packet(avpkt, MPA_MAX_CODED_FRAME_SIZE))) {

        av_log(avctx, AV_LOG_ERROR, "Error getting output packet\n");

        return ret;

    }



    init_put_bits(&s->pb, avpkt->data, avpkt->size);



    encode_frame(s, bit_alloc, padding);



    if (frame->pts != AV_NOPTS_VALUE)

        avpkt->pts = frame->pts - ff_samples_to_time_base(avctx, avctx->delay);



    avpkt->size = put_bits_count(&s->pb) / 8;

    *got_packet_ptr = 1;

    return 0;

}
