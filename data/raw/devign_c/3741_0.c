static int decode_b_picture_header(VC9Context *v)

{

    int pqindex;



    /* Prolog common to all frametypes should be done in caller */

    if (v->profile == PROFILE_SIMPLE)

    {

        av_log(v, AV_LOG_ERROR, "Found a B frame while in Simple Profile!\n");

        return FRAME_SKIPED;

    }



    v->bfraction = vc9_bfraction_lut[get_vlc2(&v->gb, vc9_bfraction_vlc.table,

                                              VC9_BFRACTION_VLC_BITS, 2)];

    if (v->bfraction < -1)

    {

        av_log(v, AV_LOG_ERROR, "Invalid BFRaction\n");

        return FRAME_SKIPED;

    }

    else if (!v->bfraction)

    {

        /* We actually have a BI frame */

        return decode_bi_picture_header(v);

    }



    /* Read the quantization stuff */

    pqindex = get_bits(&v->gb, 5);

    if (v->quantizer_mode == QUANT_FRAME_IMPLICIT)

        v->pq = pquant_table[0][pqindex];

    else

    {

        v->pq = pquant_table[v->quantizer_mode-1][pqindex];

    }

    if (pqindex < 9) v->halfpq = get_bits(&v->gb, 1);

    if (v->quantizer_mode == QUANT_FRAME_EXPLICIT)

        v->pquantizer = get_bits(&v->gb, 1);



    /* Read the MV type/mode */

    if (v->extended_mv == 1)

        v->mvrange = get_prefix(&v->gb, 0, 3);

    v->mv_mode = get_bits(&v->gb, 1);

    if (v->pq < 13)

    {

        if (!v->mv_mode)

        {

            v->mv_mode = get_bits(&v->gb, 2);

            if (v->mv_mode)

                av_log(v, AV_LOG_ERROR,

                       "mv_mode for lowquant B frame was %i\n", v->mv_mode);

        }

    }

    else

    {

        if (!v->mv_mode)

        {

            if (get_bits(&v->gb, 1))

                av_log(v, AV_LOG_ERROR,

                       "mv_mode for highquant B frame was %i\n", v->mv_mode);

        }

        v->mv_mode = 1-v->mv_mode; //To match (pq < 13) mapping

    }



    if (v->mv_mode == MV_PMODE_MIXED_MV)

    {

        if (bitplane_decoding( v->mv_type_mb_plane, v->width_mb,

                                   v->height_mb, v)<0)

            return -1;

    }



    //bitplane

    bitplane_decoding(v->direct_mb_plane, v->width_mb, v->height_mb, v);

    bitplane_decoding(v->skip_mb_plane, v->width_mb, v->height_mb, v);



    /* FIXME: what is actually chosen for B frames ? */

    v->mv_diff_vlc = &vc9_mv_diff_vlc[get_bits(&v->gb, 2)];

    v->cbpcy_vlc = &vc9_cbpcy_p_vlc[get_bits(&v->gb, 2)];

    if (v->dquant)

    {

        vop_dquant_decoding(v);

    }



    if (v->vstransform)

    {

        v->ttmbf = get_bits(&v->gb, 1);

        if (v->ttmbf)

        {

            v->ttfrm = get_bits(&v->gb, 2);

            av_log(v, AV_LOG_INFO, "Transform used: %ix%i\n",

                   (v->ttfrm & 2) ? 4 : 8, (v->ttfrm & 1) ? 4 : 8);

        }

    }

    /* Epilog should be done in caller */

    return 0;

}
