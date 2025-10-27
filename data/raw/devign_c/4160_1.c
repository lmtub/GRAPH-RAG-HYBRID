static int decode_ext_header(Wmv2Context *w){

    MpegEncContext * const s= &w->s;

    GetBitContext gb;

    int fps;

    int code;



    if(s->avctx->extradata_size<4) return -1;



    init_get_bits(&gb, s->avctx->extradata, s->avctx->extradata_size*8);



    fps                = get_bits(&gb, 5);

    s->bit_rate        = get_bits(&gb, 11)*1024;

    w->mspel_bit       = get_bits1(&gb);

    s->loop_filter     = get_bits1(&gb);

    w->abt_flag        = get_bits1(&gb);

    w->j_type_bit      = get_bits1(&gb);

    w->top_left_mv_flag= get_bits1(&gb);

    w->per_mb_rl_bit   = get_bits1(&gb);

    code               = get_bits(&gb, 3);



    if(code==0) return -1;



    s->slice_height = s->mb_height / code;



    if(s->avctx->debug&FF_DEBUG_PICT_INFO){

        av_log(s->avctx, AV_LOG_DEBUG, "fps:%d, br:%d, qpbit:%d, abt_flag:%d, j_type_bit:%d, tl_mv_flag:%d, mbrl_bit:%d, code:%d, loop_filter:%d, slices:%d\n",

        fps, s->bit_rate, w->mspel_bit, w->abt_flag, w->j_type_bit, w->top_left_mv_flag, w->per_mb_rl_bit, code, s->loop_filter,

        code);

    }

    return 0;

}
