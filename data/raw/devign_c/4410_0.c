static int advanced_decode_picture_header(VC9Context *v)

{

    static const int type_table[4] = { P_TYPE, B_TYPE, I_TYPE, BI_TYPE };

    int type, i, ret;



    if (v->interlace)

    {

        v->fcm = get_bits(&v->gb, 1);

        if (v->fcm) v->fcm = 2+get_bits(&v->gb, 1);

    }



    type = get_prefix(&v->gb, 0, 4);

    if (type > 4 || type < 0) return FRAME_SKIPED;

    v->pict_type = type_table[type];

    av_log(v->avctx, AV_LOG_INFO, "AP Frame Type: %i\n", v->pict_type);



    if (v->tfcntrflag) v->tfcntr = get_bits(&v->gb, 8);

    if (v->broadcast)

    {

        if (!v->interlace) v->rptfrm = get_bits(&v->gb, 2);

        else

        {

            v->tff = get_bits(&v->gb, 1);

            v->rff = get_bits(&v->gb, 1);

        }

    }



    if (v->panscanflag)

    {

#if 0

        for (i=0; i<v->numpanscanwin; i++)

        {

            v->topleftx[i] = get_bits(&v->gb, 16);

            v->toplefty[i] = get_bits(&v->gb, 16);

            v->bottomrightx[i] = get_bits(&v->gb, 16);

            v->bottomrighty[i] = get_bits(&v->gb, 16);

        }

#else

        skip_bits(&v->gb, 16*4*v->numpanscanwin);

#endif

    }

    v->rndctrl = get_bits(&v->gb, 1);

    v->uvsamp = get_bits(&v->gb, 1);

    if (v->finterpflag == 1) v->interpfrm = get_bits(&v->gb, 1);



    switch(v->pict_type)

    {

    case I_TYPE: if (decode_i_picture_header(v) < 0) return -1;

    case P_TYPE: if (decode_p_picture_header(v) < 0) return -1;

    case BI_TYPE:

    case B_TYPE: if (decode_b_picture_header(v) < 0) return FRAME_SKIPED;

    default: break;

    }



    /* AC/DC Syntax */

    v->transacfrm = get_bits(&v->gb, 1);

    if (v->transacfrm) v->transacfrm += get_bits(&v->gb, 1);

    if (v->pict_type == I_TYPE || v->pict_type == BI_TYPE)

    {

        v->transacfrm2 = get_bits(&v->gb, 1);

        if (v->transacfrm2) v->transacfrm2 += get_bits(&v->gb, 1);

    }

    v->transacdctab = get_bits(&v->gb, 1);

    if (v->pict_type == I_TYPE) vop_dquant_decoding(v);



    return 0;

}
