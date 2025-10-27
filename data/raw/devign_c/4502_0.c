static int rv30_decode_mb_info(RV34DecContext *r)

{

    static const int rv30_p_types[6] = { RV34_MB_SKIP, RV34_MB_P_16x16, RV34_MB_P_8x8, -1, RV34_MB_TYPE_INTRA, RV34_MB_TYPE_INTRA16x16 };

    static const int rv30_b_types[6] = { RV34_MB_SKIP, RV34_MB_B_DIRECT, RV34_MB_B_FORWARD, RV34_MB_B_BACKWARD, RV34_MB_TYPE_INTRA, RV34_MB_TYPE_INTRA16x16 };

    MpegEncContext *s = &r->s;

    GetBitContext *gb = &s->gb;

    int code = svq3_get_ue_golomb(gb);



    if(code > 11){

        av_log(s->avctx, AV_LOG_ERROR, "Incorrect MB type code\n");

        return -1;

    }

    if(code > 5){

        av_log(s->avctx, AV_LOG_ERROR, "dquant needed\n");

        code -= 6;

    }

    if(s->pict_type != AV_PICTURE_TYPE_B)

        return rv30_p_types[code];

    else

        return rv30_b_types[code];

}
