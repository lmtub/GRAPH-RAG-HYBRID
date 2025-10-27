static int decode_b_picture_secondary_header(VC9Context *v)

{

    GetBitContext *gb = &v->s.gb;

    int status;



    bitplane_decoding(&v->skip_mb_plane, v);

    if (status < 0) return -1;

#if TRACE

    if (v->mv_mode == MV_PMODE_MIXED_MV)

    {

        status = bitplane_decoding(&v->mv_type_mb_plane, v);

        if (status < 0)

            return -1;

#if TRACE

        av_log(v->s.avctx, AV_LOG_DEBUG, "MB MV Type plane encoding: "

               "Imode: %i, Invert: %i\n", status>>1, status&1);

#endif

    }



    //bitplane

    status = bitplane_decoding(&v->direct_mb_plane, v);

    if (status < 0) return -1;

#if TRACE

    av_log(v->s.avctx, AV_LOG_DEBUG, "MB Direct plane encoding: "

           "Imode: %i, Invert: %i\n", status>>1, status&1);

#endif



    av_log(v->s.avctx, AV_LOG_DEBUG, "Skip MB plane encoding: "

           "Imode: %i, Invert: %i\n", status>>1, status&1);

#endif



    /* FIXME: what is actually chosen for B frames ? */

    v->s.mv_table_index = get_bits(gb, 2); //but using vc9_ tables

    v->cbpcy_vlc = &vc9_cbpcy_p_vlc[get_bits(gb, 2)];



    if (v->dquant)

    {

        vop_dquant_decoding(v);

    }



    if (v->vstransform)

    {

        v->ttmbf = get_bits(gb, 1);

        if (v->ttmbf)

        {

            v->ttfrm = get_bits(gb, 2);

            av_log(v->s.avctx, AV_LOG_INFO, "Transform used: %ix%i\n",

                   (v->ttfrm & 2) ? 4 : 8, (v->ttfrm & 1) ? 4 : 8);

        }

    }

    /* Epilog (AC/DC syntax) should be done in caller */

    return 0;

}
