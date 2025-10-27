static int decode_b_mbs(VC9Context *v)

{

    int x, y, current_mb = 0 , last_mb = v->height_mb*v->width_mb,

        i /* MB / B postion information */;

    int direct_b_bit = 0, skip_mb_bit = 0;

    int ac_pred;

    int b_mv1 = 0, b_mv2 = 0, b_mv_type = 0;

    int mquant, mqdiff; /* MB quant stuff */

    int tt_block; /* Block transform type */

    

    for (y=0; y<v->height_mb; y++)

    {

        for (x=0; x<v->width_mb; x++)

        {

            if (v->direct_mb_plane[current_mb])

              direct_b_bit = get_bits(&v->gb, 1);

            if (1 /* Skip mode is raw */)

            {

                /* FIXME getting tired commenting */

#if 0

                skip_mb_bit = get_bits(&v->gb, n); //vlc

#endif

            }

            if (!direct_b_bit)

            {

                if (skip_mb_bit)

                {

                    /* FIXME getting tired commenting */

#if 0

                    b_mv_type = get_bits(&v->gb, n); //vlc

#endif

                }

                else

                { 

                    /* FIXME getting tired commenting */

#if 0

                    b_mv1 = get_bits(&v->gb, n); //VLC

#endif

                    if (1 /* b_mv1 isn't intra */)

                    {

                        /* FIXME: actually read it */

                        b_mv_type = 0; //vlc

                    }

                }

            }

            if (!skip_mb_bit)

            {

                if (b_mv1 != last_mb)

                {

                    GET_MQUANT();

                    if (1 /* intra mb */)

                        ac_pred = get_bits(&v->gb, 1);

                }

                else

                {

                    if (1 /* forward_mb is interpolate */)

                    {

                        /* FIXME: actually read it */

                        b_mv2 = 0; //vlc

                    }

                    if (1 /* b_mv2 isn't the last */)

                    {

                        if (1 /* intra_mb */)

                            ac_pred = get_bits(&v->gb, 1);

                        GET_MQUANT();

                    }

                }

            }

            //End1

            /* FIXME getting tired, commenting */

#if 0

            if (v->ttmbf)

                v->ttmb = get_bits(&v->gb, n); //vlc

#endif

        }

        //End2

        for (i=0; i<6; i++)

        {

            /* FIXME: process the block */

        }



        current_mb++;

    }

    return 0;

}
