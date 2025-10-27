static int standard_decode_picture_header(VC9Context *v)

{

    int status = 0;



    if (v->finterpflag) v->interpfrm = get_bits(&v->gb, 1);

    skip_bits(&v->gb, 2); //framecnt unused

    if (v->rangered) v->rangeredfrm = get_bits(&v->gb, 1);

    v->pict_type = get_bits(&v->gb, 1);

    if (v->avctx->max_b_frames && !v->pict_type)

    {

        if (get_bits(&v->gb, 1)) v->pict_type = I_TYPE;

        else v->pict_type = P_TYPE;

    }

    else v->pict_type++; //P_TYPE



    switch (v->pict_type)

    {

    case I_TYPE: status = decode_i_picture_header(v); break;

    case BI_TYPE: status = decode_b_picture_header(v); break;

    case P_TYPE: status = decode_p_picture_header(v); break;

    case B_TYPE: status = decode_b_picture_header(v); break;

    }



    if (status == FRAME_SKIPED)

    {

      av_log(v, AV_LOG_INFO, "Skipping frame...\n");

      return status;

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

   

    return 0;

}
