static void truespeech_apply_twopoint_filter(TSContext *dec, int quart)

{

    int16_t tmp[146 + 60], *ptr0, *ptr1;

    const int16_t *filter;

    int i, t, off;



    t = dec->offset2[quart];

    if(t == 127){

        memset(dec->newvec, 0, 60 * sizeof(*dec->newvec));

        return;

    }

    for(i = 0; i < 146; i++)

        tmp[i] = dec->filtbuf[i];

    off = (t / 25) + dec->offset1[quart >> 1] + 18;


    ptr0 = tmp + 145 - off;

    ptr1 = tmp + 146;

    filter = (const int16_t*)ts_order2_coeffs + (t % 25) * 2;

    for(i = 0; i < 60; i++){

        t = (ptr0[0] * filter[0] + ptr0[1] * filter[1] + 0x2000) >> 14;

        ptr0++;

        dec->newvec[i] = t;

        ptr1[i] = t;

    }

}