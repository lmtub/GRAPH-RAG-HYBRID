static void init_gain_table(COOKContext *q) {

    int i;

    q->gain_size_factor = q->samples_per_channel/8;

    for (i=0 ; i<23 ; i++) {

        q->gain_table[i] = pow((double)q->pow2tab[i+52] ,

                               (1.0/(double)q->gain_size_factor));

    }

    memset(&q->gain_copy, 0, sizeof(COOKgain));

    memset(&q->gain_current, 0, sizeof(COOKgain));

    memset(&q->gain_now, 0, sizeof(COOKgain));

    memset(&q->gain_previous, 0, sizeof(COOKgain));

}
