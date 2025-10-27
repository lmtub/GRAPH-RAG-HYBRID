static int encode_superframe(AVCodecContext *avctx, AVPacket *avpkt,

                             const AVFrame *frame, int *got_packet_ptr)

{

    WMACodecContext *s = avctx->priv_data;

    int i, total_gain, ret, error;



    s->block_len_bits= s->frame_len_bits; //required by non variable block len

    s->block_len = 1 << s->block_len_bits;



    apply_window_and_mdct(avctx, frame);



    if (s->ms_stereo) {

        float a, b;

        int i;



        for(i = 0; i < s->block_len; i++) {

            a = s->coefs[0][i]*0.5;

            b = s->coefs[1][i]*0.5;

            s->coefs[0][i] = a + b;

            s->coefs[1][i] = a - b;





    if ((ret = ff_alloc_packet2(avctx, avpkt, 2 * MAX_CODED_SUPERFRAME_SIZE)) < 0)

        return ret;



    total_gain= 128;

    for(i=64; i; i>>=1){

        error = encode_frame(s, s->coefs, avpkt->data, avpkt->size,

                                 total_gain - i);

        if(error<=0)

            total_gain-= i;




    while(total_gain <= 128 && error > 0)

        error = encode_frame(s, s->coefs, avpkt->data, avpkt->size, total_gain++);






    av_assert0((put_bits_count(&s->pb) & 7) == 0);

    i= avctx->block_align - (put_bits_count(&s->pb)+7)/8;

    av_assert0(i>=0);

    while(i--)

        put_bits(&s->pb, 8, 'N');



    flush_put_bits(&s->pb);

    av_assert0(put_bits_ptr(&s->pb) - s->pb.buf == avctx->block_align);



    if (frame->pts != AV_NOPTS_VALUE)

        avpkt->pts = frame->pts - ff_samples_to_time_base(avctx, avctx->delay);



    avpkt->size = avctx->block_align;

    *got_packet_ptr = 1;

    return 0;
