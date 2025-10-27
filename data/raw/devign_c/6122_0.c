static int g726_encode_frame(AVCodecContext *avctx, AVPacket *avpkt,

                             const AVFrame *frame, int *got_packet_ptr)

{

    G726Context *c = avctx->priv_data;

    const int16_t *samples = (const int16_t *)frame->data[0];

    PutBitContext pb;

    int i, ret, out_size;



    out_size = (frame->nb_samples * c->code_size + 7) / 8;

    if ((ret = ff_alloc_packet2(avctx, avpkt, out_size)))

        return ret;

    init_put_bits(&pb, avpkt->data, avpkt->size);



    for (i = 0; i < frame->nb_samples; i++)

        put_bits(&pb, c->code_size, g726_encode(c, *samples++));



    flush_put_bits(&pb);



    avpkt->size = out_size;

    *got_packet_ptr = 1;

    return 0;

}
