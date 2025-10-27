static int decode_p_mbs(VC9Context *v)

{

    int x, y, current_mb = 0, i; /* MB/Block Position info */

    int skip_mb_bit = 0, cbpcy; /* MB/B skip */

    int hybrid_pred, ac_pred; /* Prediction types */

    int mb_has_coeffs = 1  /* last_flag */, mb_is_intra;

    int dmv_x, dmv_y; /* Differential MV components */

    int mv_mode_bit = 0; /* mv_mode_bit: 1MV=0, 4MV=0 */

    int mqdiff, mquant; /* MB quantization */

    int tt_block; /* MB Transform type */

    static const int size_table[6] = { 0, 2, 3, 4, 5, 8 },

        offset_table[6] = { 0, 1, 3, 7, 15, 31 };

    int k_x, k_y; /* Long MV fixed bitlength */

    int hpel_flag, intra_flag; /* Some MB properties */

    int index, index1; /* LUT indices */

    int val, sign;



    /* Select ttmb table depending on pq */

    if (v->pq < 5) v->ttmb_vlc = &vc9_ttmb_vlc[0];

    else if (v->pq < 13) v->ttmb_vlc = &vc9_ttmb_vlc[1];

    else v->ttmb_vlc = &vc9_ttmb_vlc[2];



    /* Select proper long MV range */

    switch (v->mvrange)

    {

    case 1: k_x = 10; k_y = 9; break;

    case 2: k_x = 12; k_y = 10; break;

    case 3: k_x = 13; k_y = 11; break;

    default: /*case 0 too */ k_x = 9; k_y = 8; break;

    }



    hpel_flag = v->mv_mode & 1; //MV_PMODE is HPEL

    k_x -= hpel_flag;

    k_y -= hpel_flag;



    for (y=0; y<v->height_mb; y++)

    {

        for (x=0; x<v->width_mb; x++)

        {

            if (v->mv_type_mb_plane[current_mb])

                mv_mode_bit = get_bits(&v->gb, 1);

            if (0) //skipmb is rawmode

                skip_mb_bit = get_bits(&v->gb, 1);

            if (!mv_mode_bit) /* 1MV mode */

            {

                if (!v->skip_mb_plane[current_mb])

                {

                    GET_MVDATA();



                    /* hybrid mv pred, 8.3.5.3.4 */

                    if (v->mv_mode == MV_PMODE_1MV ||

                        v->mv_mode == MV_PMODE_MIXED_MV)

                        hybrid_pred = get_bits(&v->gb, 1);

                    if (mb_is_intra && !mb_has_coeffs)

                    {

                        GET_MQUANT();

                        ac_pred = get_bits(&v->gb, 1);

                    }

                    else if (mb_has_coeffs)

                    {

                        if (mb_is_intra) ac_pred = get_bits(&v->gb, 1);

                        cbpcy = get_vlc2(&v->gb, v->cbpcy_vlc->table,

                                         VC9_CBPCY_P_VLC_BITS, 2);

                        GET_MQUANT();

                    }

                    if (!v->ttmbf)

                        v->ttfrm = get_vlc2(&v->gb, v->ttmb_vlc->table,

                                            VC9_TTMB_VLC_BITS, 2);

                    //Decode blocks from that mb wrt cbpcy

                }

                else //Skipped

                {

                    /* hybrid mv pred, 8.3.5.3.4 */

                    if (v->mv_mode == MV_PMODE_1MV ||

                        v->mv_mode == MV_PMODE_MIXED_MV)

                        hybrid_pred = get_bits(&v->gb, 1);

                }

            } //1MV mode

            else //4MV mode

            {

              if (!v->skip_mb_plane[current_mb] /* unskipped MB */)

                {

                    cbpcy = get_vlc2(&v->gb, v->cbpcy_vlc->table,

                                     VC9_CBPCY_P_VLC_BITS, 2);

                    for (i=0; i<4; i++) //For all 4 Y blocks

                    {

                        if (cbpcy & (1<<6) /* cbpcy set for this block */)

                        {

                            GET_MVDATA();

                        }

                        if (v->mv_mode == MV_PMODE_MIXED_MV /* Hybrid pred */)

                            hybrid_pred = get_bits(&v->gb, 1);

                        GET_MQUANT();

                        if (mb_is_intra /* One of the 4 blocks is intra */ &&

                            index /* non-zero pred for that block */)

                            ac_pred = get_bits(&v->gb, 1);

                        if (!v->ttmbf)

                            tt_block = get_vlc2(&v->gb, v->ttmb_vlc->table,

                                                VC9_TTMB_VLC_BITS, 2);

            

                        /* TODO: Process blocks wrt cbpcy */

            

                        /* Prepare cbpcy for next block */

                        cbpcy <<= 1;

                    }

                }

                else //Skipped MB

                {

                    for (i=0; i<4; i++) //All 4 Y blocks

                    {

                        if (v->mv_mode == MV_PMODE_MIXED_MV /* Hybrid pred */)

                            hybrid_pred = get_bits(&v->gb, 1);

                        

                        /* FIXME: do something */

                    }

                }

            }

        }

        current_mb++;

    }

    return 0;

}
