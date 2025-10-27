static int rv34_decode_cbp(GetBitContext *gb, RV34VLC *vlc, int table)

{

    int pattern, code, cbp=0;

    int ones;

    static const int cbp_masks[3] = {0x100000, 0x010000, 0x110000};

    static const int shifts[4] = { 0, 2, 8, 10 };

    const int *curshift = shifts;

    int i, t, mask;



    code = get_vlc2(gb, vlc->cbppattern[table].table, 9, 2);

    pattern = code & 0xF;

    code >>= 4;



    ones = rv34_count_ones[pattern];



    for(mask = 8; mask; mask >>= 1, curshift++){

        if(pattern & mask)

            cbp |= get_vlc2(gb, vlc->cbp[table][ones].table, vlc->cbp[table][ones].bits, 1) << curshift[0];

    }



    for(i = 0; i < 4; i++){

        t = modulo_three_table[code][i];

        if(t == 1)

            cbp |= cbp_masks[get_bits1(gb)] << i;

        if(t == 2)

            cbp |= cbp_masks[2] << i;

    }

    return cbp;

}
