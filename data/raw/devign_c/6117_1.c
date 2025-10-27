static int encode_frame(AVCodecContext *avctx, AVPacket *avpkt,
                        const AVFrame *frame, int *got_packet_ptr)
{
    DCAEncContext *c = avctx->priv_data;
    const int32_t *samples;
    int ret, i;
    if ((ret = ff_alloc_packet2(avctx, avpkt, c->frame_size , 0)) < 0)
        return ret;
    samples = (const int32_t *)frame->data[0];
    subband_transform(c, samples);
    if (c->lfe_channel)
        lfe_downsample(c, samples);
    calc_masking(c, samples);
    find_peaks(c);
    assign_bits(c);
    calc_scales(c);
    quantize_all(c);
    shift_history(c, samples);
    init_put_bits(&c->pb, avpkt->data, avpkt->size);
    put_frame_header(c);
    put_primary_audio_header(c);
    for (i = 0; i < SUBFRAMES; i++)
        put_subframe(c, i);
    flush_put_bits(&c->pb);
    avpkt->pts      = frame->pts;
    avpkt->duration = ff_samples_to_time_base(avctx, frame->nb_samples);
    avpkt->size     = c->frame_size + 1;
    *got_packet_ptr = 1;
    return 0;
}