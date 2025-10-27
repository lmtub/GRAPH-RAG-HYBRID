static int decode_p_picture_primary_header(VC9Context *v)

{

    /* INTERFRM, FRMCNT, RANGEREDFRM read in caller */

    GetBitContext *gb = &v->s.gb;

    int lowquant, pqindex, status = 0;



    pqindex = get_bits(gb, 5);

    if (v->quantizer_mode == QUANT_FRAME_IMPLICIT)

        v->pq = pquant_table[0][pqindex];

    else

    {

        v->pq = pquant_table[v->quantizer_mode-1][pqindex];

    }

    if (pqindex < 9) v->halfpq = get_bits(gb, 1);

    if (v->quantizer_mode == QUANT_FRAME_EXPLICIT)

        v->pquantizer = get_bits(gb, 1);

    av_log(v->s.avctx, AV_LOG_DEBUG, "P Frame: QP=%i (+%i/2)\n",

           v->pq, v->halfpq);

    if (v->extended_mv == 1) v->mvrange = get_prefix(gb, 0, 3);

#if HAS_ADVANCED_PROFILE

    if (v->profile > PROFILE_MAIN)

    {

        if (v->postprocflag) v->postproc = get_bits(gb, 1);

    }

    else

#endif

        if (v->multires) v->respic = get_bits(gb, 2);

    lowquant = (v->pquantizer>12) ? 0 : 1;

    v->mv_mode = mv_pmode_table[lowquant][get_prefix(gb, 1, 4)];

    if (v->mv_mode == MV_PMODE_INTENSITY_COMP)

    {

        v->mv_mode2 = mv_pmode_table[lowquant][get_prefix(gb, 1, 3)];

        v->lumscale = get_bits(gb, 6);

        v->lumshift = get_bits(gb, 6);

    }

    return 0;

}
