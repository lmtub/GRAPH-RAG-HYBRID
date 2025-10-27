int decode_luma_intra_block(VC9Context *v, int mquant)

{

    GetBitContext *gb = &v->s.gb;

    int dcdiff;



    dcdiff = get_vlc2(gb, v->luma_dc_vlc->table,

                      DC_VLC_BITS, 2);

    if (dcdiff)

    {

        if (dcdiff == 119 /* ESC index value */)

        {

            /* TODO: Optimize */

            if (mquant == 1) dcdiff = get_bits(gb, 10);

            else if (mquant == 2) dcdiff = get_bits(gb, 9);

            else dcdiff = get_bits(gb, 8);

        }

        else

        {

            if (mquant == 1)

                dcdiff = (dcdiff<<2) + get_bits(gb, 2) - 3;

            else if (mquant == 2)

                dcdiff = (dcdiff<<1) + get_bits(gb, 1) - 1;

        }

        if (get_bits(gb, 1))

            dcdiff = -dcdiff;

    }

    /* FIXME: 8.1.1.15, p(1)13, coeff scaling for Adv Profile */



    return 0;

}
