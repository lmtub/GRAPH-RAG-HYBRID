static int bands_dist(OpusPsyContext *s, CeltFrame *f, float *total_dist)

{

    int i, tdist = 0.0f;

    OpusRangeCoder dump;



    ff_opus_rc_enc_init(&dump);

    ff_celt_enc_bitalloc(f, &dump);



    for (i = 0; i < CELT_MAX_BANDS; i++) {

        float bits = 0.0f;

        float dist = f->pvq->band_cost(f->pvq, f, &dump, i, &bits, s->lambda);

        tdist += dist;

    }



    *total_dist = tdist;



    return 0;

}
