static av_always_inline void get_mvdata_interlaced(VC1Context *v, int *dmv_x,

                                                   int *dmv_y, int *pred_flag)

{

    int index, index1;

    int extend_x = 0, extend_y = 0;

    GetBitContext *gb = &v->s.gb;

    int bits, esc;

    int val, sign;

    const int* offs_tab;



    if (v->numref) {

        bits = VC1_2REF_MVDATA_VLC_BITS;

        esc  = 125;

    } else {

        bits = VC1_1REF_MVDATA_VLC_BITS;

        esc  = 71;

    }

    switch (v->dmvrange) {

    case 1:

        extend_x = 1;

        break;

    case 2:

        extend_y = 1;

        break;

    case 3:

        extend_x = extend_y = 1;

        break;

    }

    index = get_vlc2(gb, v->imv_vlc->table, bits, 3);

    if (index == esc) {

        *dmv_x = get_bits(gb, v->k_x);

        *dmv_y = get_bits(gb, v->k_y);

        if (v->numref) {

            *pred_flag = *dmv_y & 1;

            *dmv_y     = (*dmv_y + *pred_flag) >> 1;

        }

    }

    else {

        if (extend_x)

            offs_tab = offset_table2;

        else

            offs_tab = offset_table1;

        index1 = (index + 1) % 9;

        if (index1 != 0) {

            val    = get_bits(gb, index1 + extend_x);

            sign   = 0 -(val & 1);

            *dmv_x = (sign ^ ((val >> 1) + offs_tab[index1])) - sign;

        } else

            *dmv_x = 0;

        if (extend_y)

            offs_tab = offset_table2;

        else

            offs_tab = offset_table1;

        index1 = (index + 1) / 9;

        if (index1 > v->numref) {

            val    = get_bits(gb, (index1 + (extend_y << v->numref)) >> v->numref);

            sign   = 0 - (val & 1);

            *dmv_y = (sign ^ ((val >> 1) + offs_tab[index1 >> v->numref])) - sign;

        } else

            *dmv_y = 0;

        if (v->numref)

            *pred_flag = index1 & 1;

    }

}
