static void put_subframe(DCAEncContext *c, int subframe)

{

    int i, band, ss, ch;



    /* Subsubframes count */

    put_bits(&c->pb, 2, SUBSUBFRAMES -1);



    /* Partial subsubframe sample count: dummy */

    put_bits(&c->pb, 3, 0);



    /* Prediction mode: no ADPCM, in each channel and subband */

    for (ch = 0; ch < c->fullband_channels; ch++)

        for (band = 0; band < DCAENC_SUBBANDS; band++)

            put_bits(&c->pb, 1, 0);



    /* Prediction VQ address: not transmitted */

    /* Bit allocation index */

    for (ch = 0; ch < c->fullband_channels; ch++)

        for (band = 0; band < DCAENC_SUBBANDS; band++)

            put_bits(&c->pb, 5, c->abits[band][ch]);



    if (SUBSUBFRAMES > 1) {

        /* Transition mode: none for each channel and subband */

        for (ch = 0; ch < c->fullband_channels; ch++)

            for (band = 0; band < DCAENC_SUBBANDS; band++)

                put_bits(&c->pb, 1, 0); /* codebook A4 */

    }



    /* Scale factors */

    for (ch = 0; ch < c->fullband_channels; ch++)

        for (band = 0; band < DCAENC_SUBBANDS; band++)

            put_bits(&c->pb, 7, c->scale_factor[band][ch]);



    /* Joint subband scale factor codebook select: not transmitted */

    /* Scale factors for joint subband coding: not transmitted */

    /* Stereo down-mix coefficients: not transmitted */

    /* Dynamic range coefficient: not transmitted */

    /* Stde information CRC check word: not transmitted */

    /* VQ encoded high frequency subbands: not transmitted */



    /* LFE data: 8 samples and scalefactor */

    if (c->lfe_channel) {

        for (i = 0; i < DCA_LFE_SAMPLES; i++)

            put_bits(&c->pb, 8, quantize_value(c->downsampled_lfe[i], c->lfe_quant) & 0xff);

        put_bits(&c->pb, 8, c->lfe_scale_factor);

    }



    /* Audio data (subsubframes) */

    for (ss = 0; ss < SUBSUBFRAMES ; ss++)

        for (ch = 0; ch < c->fullband_channels; ch++)

            for (band = 0; band < DCAENC_SUBBANDS; band++)

                    put_subframe_samples(c, ss, band, ch);



    /* DSYNC */

    put_bits(&c->pb, 16, 0xffff);

}
