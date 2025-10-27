static int decode_exponents(GetBitContext *gbc, int exp_strategy, int ngrps,

                            uint8_t absexp, int8_t *dexps)

{

    int i, j, grp, group_size;

    int dexp[256];

    int expacc, prevexp;



    /* unpack groups */

    group_size = exp_strategy + (exp_strategy == EXP_D45);

    for(grp=0,i=0; grp<ngrps; grp++) {

        expacc = get_bits(gbc, 7);

        dexp[i++] = ungroup_3_in_7_bits_tab[expacc][0];

        dexp[i++] = ungroup_3_in_7_bits_tab[expacc][1];

        dexp[i++] = ungroup_3_in_7_bits_tab[expacc][2];

    }



    /* convert to absolute exps and expand groups */

    prevexp = absexp;

    for(i=0,j=0; i<ngrps*3; i++) {

        prevexp += dexp[i] - 2;

        if (prevexp < 0 || prevexp > 24)

            return -1;

        switch (group_size) {

            case 4: dexps[j++] = prevexp;

                    dexps[j++] = prevexp;

            case 2: dexps[j++] = prevexp;

            case 1: dexps[j++] = prevexp;

        }

    }

    return 0;

}
