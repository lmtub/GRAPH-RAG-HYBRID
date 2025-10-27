static float quantize_band_cost(struct AACEncContext *s, const float *in,

                                const float *scaled, int size, int scale_idx,

                                int cb, const float lambda, const float uplim,

                                int *bits)

{

    const float IQ = ff_aac_pow2sf_tab[200 + scale_idx - SCALE_ONE_POS + SCALE_DIV_512];

    const float  Q = ff_aac_pow2sf_tab[200 - scale_idx + SCALE_ONE_POS - SCALE_DIV_512];

    const float CLIPPED_ESCAPE = 165140.0f*IQ;

    int i, j, k;

    float cost = 0;

    const int dim = cb < FIRST_PAIR_BT ? 4 : 2;

    int resbits = 0;

#ifndef USE_REALLY_FULL_SEARCH

    const float  Q34 = sqrtf(Q * sqrtf(Q));

    const int range  = aac_cb_range[cb];

    const int maxval = aac_cb_maxval[cb];

    int offs[4];

#endif /* USE_REALLY_FULL_SEARCH */



    if (!cb) {

        for (i = 0; i < size; i++)

            cost += in[i]*in[i];

        if (bits)

            *bits = 0;

        return cost * lambda;

    }

#ifndef USE_REALLY_FULL_SEARCH

    offs[0] = 1;

    for (i = 1; i < dim; i++)

        offs[i] = offs[i-1]*range;

    quantize_bands(s->qcoefs, in, scaled, size, Q34, !IS_CODEBOOK_UNSIGNED(cb), maxval);

#endif /* USE_REALLY_FULL_SEARCH */

    for (i = 0; i < size; i += dim) {

        float mincost;

        int minidx  = 0;

        int minbits = 0;

        const float *vec;

#ifndef USE_REALLY_FULL_SEARCH

        int (*quants)[2] = &s->qcoefs[i];

        mincost = 0.0f;

        for (j = 0; j < dim; j++)

            mincost += in[i+j]*in[i+j];

        minidx = IS_CODEBOOK_UNSIGNED(cb) ? 0 : 40;

        minbits = ff_aac_spectral_bits[cb-1][minidx];

        mincost = mincost * lambda + minbits;

        for (j = 0; j < (1<<dim); j++) {

            float rd = 0.0f;

            int curbits;

            int curidx = IS_CODEBOOK_UNSIGNED(cb) ? 0 : 40;

            int same   = 0;

            for (k = 0; k < dim; k++) {

                if ((j & (1 << k)) && quants[k][0] == quants[k][1]) {

                    same = 1;

                    break;

                }

            }

            if (same)

                continue;

            for (k = 0; k < dim; k++)

                curidx += quants[k][!!(j & (1 << k))] * offs[dim - 1 - k];

            curbits =  ff_aac_spectral_bits[cb-1][curidx];

            vec     = &ff_aac_codebook_vectors[cb-1][curidx*dim];

#else

        mincost = INFINITY;

        vec = ff_aac_codebook_vectors[cb-1];

        for (j = 0; j < ff_aac_spectral_sizes[cb-1]; j++, vec += dim) {

            float rd = 0.0f;

            int curbits = ff_aac_spectral_bits[cb-1][j];

#endif /* USE_REALLY_FULL_SEARCH */

            if (IS_CODEBOOK_UNSIGNED(cb)) {

                for (k = 0; k < dim; k++) {

                    float t = fabsf(in[i+k]);

                    float di;

                    if (vec[k] == 64.0f) { //FIXME: slow

                        //do not code with escape sequence small values

                        if (t < 39.0f*IQ) {

                            rd = INFINITY;

                            break;

                        }

                        if (t >= CLIPPED_ESCAPE) {

                            di = t - CLIPPED_ESCAPE;

                            curbits += 21;

                        } else {

                            int c = av_clip(quant(t, Q), 0, 8191);

                            di = t - c*cbrtf(c)*IQ;

                            curbits += av_log2(c)*2 - 4 + 1;

                        }

                    } else {

                        di = t - vec[k]*IQ;

                    }

                    if (vec[k] != 0.0f)

                        curbits++;

                    rd += di*di;

                }

            } else {

                for (k = 0; k < dim; k++) {

                    float di = in[i+k] - vec[k]*IQ;

                    rd += di*di;

                }

            }

            rd = rd * lambda + curbits;

            if (rd < mincost) {

                mincost = rd;

                minidx  = j;

                minbits = curbits;

            }

        }

        cost    += mincost;

        resbits += minbits;

        if (cost >= uplim)

            return uplim;

    }



    if (bits)

        *bits = resbits;

    return cost;

}
