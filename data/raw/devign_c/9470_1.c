static int decode_b_mbs(VC9Context *v)

{

    MpegEncContext *s = &v->s;

    GetBitContext *gb = &v->s.gb;

    int current_mb = 0, i /* MB / B postion information */;

    int b_mv_type = BMV_TYPE_BACKWARD;

    int mquant, mqdiff; /* MB quant stuff */

    int ttmb; /* MacroBlock transform type */

    

    static const int size_table[6] = { 0, 2, 3, 4, 5, 8 },

        offset_table[6] = { 0, 1, 3, 7, 15, 31 };

    int mb_has_coeffs = 1; /* last_flag */

    int dmv1_x, dmv1_y, dmv2_x, dmv2_y; /* Differential MV components */

    int k_x, k_y; /* Long MV fixed bitlength */

    int hpel_flag; /* Some MB properties */

    int index, index1; /* LUT indices */

    int val, sign; /* MVDATA temp values */

    

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



    /* Select ttmb table depending on pq */

    if (v->pq < 5) v->ttmb_vlc = &vc9_ttmb_vlc[0];

    else if (v->pq < 13) v->ttmb_vlc = &vc9_ttmb_vlc[1];

    else v->ttmb_vlc = &vc9_ttmb_vlc[2];



    for (s->mb_y=0; s->mb_y<s->mb_height; s->mb_y++)

    {

        for (s->mb_x=0; s->mb_x<s->mb_width; s->mb_x++)

        {

            if (v->direct_mb_plane.is_raw)

                v->direct_mb_plane.data[current_mb] = get_bits(gb, 1);

            if (v->skip_mb_plane.is_raw)

                v->skip_mb_plane.data[current_mb] = get_bits(gb, 1);

            

            if (!v->direct_mb_plane.data[current_mb])

            {

                if (v->skip_mb_plane.data[current_mb])

                {

                    b_mv_type = decode012(gb);

                    if (v->bfraction > 420 /*1/2*/ &&

                        b_mv_type < 3) b_mv_type = 1-b_mv_type;

                }

                else

                { 

                    GET_MVDATA(dmv1_x, dmv1_y);

                    if (!s->mb_intra /* b_mv1 tells not intra */)

                    {

                        b_mv_type = decode012(gb);

                        if (v->bfraction > 420 /*1/2*/ &&

                            b_mv_type < 3) b_mv_type = 1-b_mv_type;

                    }

                }

            }

            if (!v->skip_mb_plane.data[current_mb])

            {

                if (mb_has_coeffs /* BMV1 == "last" */)

                {

                    GET_MQUANT();

                    if (s->mb_intra /* intra mb */)

                        s->ac_pred = get_bits(gb, 1);

                }

                else

                {

                    /* if bmv1 tells MVs are interpolated */

                    if (b_mv_type == BMV_TYPE_INTERPOLATED)

                    {

                        GET_MVDATA(dmv2_x, dmv2_y);

                    }

                    /* GET_MVDATA has reset some stuff */

                    if (mb_has_coeffs /* b_mv2 == "last" */)

                    {

                        if (s->mb_intra /* intra_mb */)

                            s->ac_pred = get_bits(gb, 1);

                        GET_MQUANT();

                    }

                }

            }

            //End1

            if (v->ttmbf)

                ttmb = get_vlc2(gb, v->ttmb_vlc->table,

                                   VC9_TTMB_VLC_BITS, 12);



            //End2

            for (i=0; i<6; i++)

            {

                /* FIXME: process the block */

            }



            current_mb++;

        }

    }

    return 0;

}
