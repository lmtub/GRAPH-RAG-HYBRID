static int encode_superframe(AVCodecContext *avctx,

                            unsigned char *buf, int buf_size, void *data){

    WMACodecContext *s = avctx->priv_data;

    const short *samples = data;

    int i, total_gain;



    s->block_len_bits= s->frame_len_bits; //required by non variable block len

    s->block_len = 1 << s->block_len_bits;



    apply_window_and_mdct(avctx, samples, avctx->frame_size);



    if (s->ms_stereo) {

        float a, b;

        int i;



        for(i = 0; i < s->block_len; i++) {

            a = s->coefs[0][i]*0.5;

            b = s->coefs[1][i]*0.5;

            s->coefs[0][i] = a + b;

            s->coefs[1][i] = a - b;

        }

    }



    if (buf_size < 2 * MAX_CODED_SUPERFRAME_SIZE) {

        av_log(avctx, AV_LOG_ERROR, "output buffer size is too small\n");

        return AVERROR(EINVAL);

    }



#if 1

    total_gain= 128;

    for(i=64; i; i>>=1){

        int error= encode_frame(s, s->coefs, buf, buf_size, total_gain-i);

        if(error<0)

            total_gain-= i;

    }

#else

    total_gain= 90;

    best= encode_frame(s, s->coefs, buf, buf_size, total_gain);

    for(i=32; i; i>>=1){

        int scoreL= encode_frame(s, s->coefs, buf, buf_size, total_gain-i);

        int scoreR= encode_frame(s, s->coefs, buf, buf_size, total_gain+i);

        av_log(NULL, AV_LOG_ERROR, "%d %d %d (%d)\n", scoreL, best, scoreR, total_gain);

        if(scoreL < FFMIN(best, scoreR)){

            best = scoreL;

            total_gain -= i;

        }else if(scoreR < best){

            best = scoreR;

            total_gain += i;

        }

    }

#endif



    if ((i = encode_frame(s, s->coefs, buf, buf_size, total_gain)) >= 0) {

        av_log(avctx, AV_LOG_ERROR, "required frame size too large. please "

               "use a higher bit rate.\n");

        return AVERROR(EINVAL);

    }

    assert((put_bits_count(&s->pb) & 7) == 0);

    while (i++)

        put_bits(&s->pb, 8, 'N');



    flush_put_bits(&s->pb);

    return s->block_align;

}
