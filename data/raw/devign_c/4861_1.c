static int advanced_decode_picture_primary_header(VC9Context *v)

{

    GetBitContext *gb = &v->s.gb;

    static const int type_table[4] = { P_TYPE, B_TYPE, I_TYPE, BI_TYPE };

    int type, i;



    if (v->interlace)

    {

        v->fcm = get_bits(gb, 1);

        if (v->fcm) v->fcm = 2+get_bits(gb, 1);

    }



    type = get_prefix(gb, 0, 4);

    if (type > 4 || type < 0) return FRAME_SKIPED;

    v->s.pict_type = type_table[type];

    av_log(v->s.avctx, AV_LOG_INFO, "AP Frame Type: %i\n", v->s.pict_type);



    if (v->tfcntrflag) v->tfcntr = get_bits(gb, 8);

    if (v->broadcast)

    {

        if (!v->interlace) v->rptfrm = get_bits(gb, 2);

        else

        {

            v->tff = get_bits(gb, 1);

            v->rff = get_bits(gb, 1);

        }

    }



    if (v->panscanflag)

    {

#if 0

        for (i=0; i<v->numpanscanwin; i++)

        {

            v->topleftx[i] = get_bits(gb, 16);

            v->toplefty[i] = get_bits(gb, 16);

            v->bottomrightx[i] = get_bits(gb, 16);

            v->bottomrighty[i] = get_bits(gb, 16);

        }

#else

        skip_bits(gb, 16*4*v->numpanscanwin);

#endif

    }

    v->s.no_rounding = !get_bits(gb, 1);

    v->uvsamp = get_bits(gb, 1);

    if (v->finterpflag == 1) v->interpfrm = get_bits(gb, 1);



    switch(v->s.pict_type)

    {

    case I_TYPE: if (decode_i_picture_header(v) < 0) return -1;

    case P_TYPE: if (decode_p_picture_primary_header(v) < 0) return -1;

    case BI_TYPE:

    case B_TYPE: if (decode_b_picture_primary_header(v) < 0) return FRAME_SKIPED;

    default: break;

    }

    return 0;

}
