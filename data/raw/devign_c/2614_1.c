static int mpeg1_decode_picture(AVCodecContext *avctx, 

                                const uint8_t *buf, int buf_size)

{

    Mpeg1Context *s1 = avctx->priv_data;

    MpegEncContext *s = &s1->mpeg_enc_ctx;

    int ref, f_code, vbv_delay;



    if(mpeg_decode_postinit(s->avctx) < 0) 

       return -2;



    init_get_bits(&s->gb, buf, buf_size*8);



    ref = get_bits(&s->gb, 10); /* temporal ref */

    s->pict_type = get_bits(&s->gb, 3);



    vbv_delay= get_bits(&s->gb, 16);

    if (s->pict_type == P_TYPE || s->pict_type == B_TYPE) {

        s->full_pel[0] = get_bits1(&s->gb);

        f_code = get_bits(&s->gb, 3);

        if (f_code == 0)

            return -1;

        s->mpeg_f_code[0][0] = f_code;

        s->mpeg_f_code[0][1] = f_code;

    }

    if (s->pict_type == B_TYPE) {

        s->full_pel[1] = get_bits1(&s->gb);

        f_code = get_bits(&s->gb, 3);

        if (f_code == 0)

            return -1;

        s->mpeg_f_code[1][0] = f_code;

        s->mpeg_f_code[1][1] = f_code;

    }

    s->current_picture.pict_type= s->pict_type;

    s->current_picture.key_frame= s->pict_type == I_TYPE;

    

//    if(avctx->debug & FF_DEBUG_PICT_INFO)

//        av_log(avctx, AV_LOG_DEBUG, "vbv_delay %d, ref %d\n", vbv_delay, ref);

    

    s->y_dc_scale = 8;

    s->c_dc_scale = 8;

    s->first_slice = 1;

    return 0;

}
