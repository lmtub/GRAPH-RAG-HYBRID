void nelly_decode_block(NellyMoserDecodeContext *s, unsigned char block[NELLY_BLOCK_LEN], float audio[NELLY_SAMPLES])

{

    int i,j;

    float buf[NELLY_FILL_LEN], pows[NELLY_FILL_LEN];

    float *aptr, *bptr, *pptr, val, pval;

    int bits[NELLY_BUF_LEN];

    unsigned char v;



    init_get_bits(&s->gb, block, NELLY_BLOCK_LEN * 8);



    bptr = buf;

    pptr = pows;

    val = nelly_init_table[get_bits(&s->gb, 6)];

    for (i=0 ; i<NELLY_BANDS ; i++) {

        if (i > 0)

            val += nelly_delta_table[get_bits(&s->gb, 5)];

        pval = pow(2, val/2048);

        for (j = 0; j < nelly_band_sizes_table[i]; j++) {

            *bptr++ = val;

            *pptr++ = pval;

        }



    }



    get_sample_bits(buf, bits);



    for (i = 0; i < 2; i++) {

        aptr = audio + i * NELLY_BUF_LEN;



        init_get_bits(&s->gb, block, NELLY_BLOCK_LEN * 8);

        skip_bits(&s->gb, NELLY_HEADER_BITS + i*NELLY_DETAIL_BITS);



        for (j = 0; j < NELLY_FILL_LEN; j++) {

            if (bits[j] <= 0) {

                aptr[j] = M_SQRT1_2*pows[j];

                if (av_random(&s->random_state) & 1)

                    aptr[j] *= -1.0;

            } else {

                v = get_bits(&s->gb, bits[j]);

                aptr[j] = dequantization_table[(1<<bits[j])-1+v]*pows[j];

            }

        }

        memset(&aptr[NELLY_FILL_LEN], 0,

               (NELLY_BUF_LEN - NELLY_FILL_LEN) * sizeof(float));



        s->imdct_ctx.fft.imdct_calc(&s->imdct_ctx, s->imdct_out,

                                    aptr, s->imdct_tmp);

        /* XXX: overlapping and windowing should be part of a more

           generic imdct function */

        for(j = 0; j < NELLY_BUF_LEN / 2; j++) {

            aptr[j] = s->imdct_out[j + NELLY_BUF_LEN + NELLY_BUF_LEN / 2];

            aptr[j + NELLY_BUF_LEN / 2] = s->imdct_out[j];

        }

        overlap_and_window(s, s->state, aptr);

    }

}
