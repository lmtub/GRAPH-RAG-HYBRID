static av_always_inline float quantize_and_encode_band_cost_template(

                                struct AACEncContext *s,

                                PutBitContext *pb, const float *in,

                                const float *scaled, int size, int scale_idx,

                                int cb, const float lambda, const float uplim,

                                int *bits, int BT_ZERO, int BT_UNSIGNED,

                                int BT_PAIR, int BT_ESC)

{

    const float IQ = ff_aac_pow2sf_tab[200 + scale_idx - SCALE_ONE_POS + SCALE_DIV_512];

    const float  Q = ff_aac_pow2sf_tab[200 - scale_idx + SCALE_ONE_POS - SCALE_DIV_512];

    const float CLIPPED_ESCAPE = 165140.0f*IQ;

    int i, j, k;

    float cost = 0;

    const int dim = BT_PAIR ? 2 : 4;

    int resbits = 0;

    const float  Q34 = sqrtf(Q * sqrtf(Q));

    const int range  = aac_cb_range[cb];

    const int maxval = aac_cb_maxval[cb];

    int off;



    if (BT_ZERO) {

        for (i = 0; i < size; i++)

            cost += in[i]*in[i];

        if (bits)

            *bits = 0;

        return cost * lambda;

    }

    if (!scaled) {

        abs_pow34_v(s->scoefs, in, size);

        scaled = s->scoefs;

    }

    quantize_bands(s->qcoefs, in, scaled, size, Q34, !BT_UNSIGNED, maxval);

    if (BT_UNSIGNED) {

        off = 0;

    } else {

        off = maxval;

    }

    for (i = 0; i < size; i += dim) {

        const float *vec;

        int *quants = s->qcoefs + i;

        int curidx = 0;

        int curbits;

        float rd = 0.0f;

        for (j = 0; j < dim; j++) {

            curidx *= range;

            curidx += quants[j] + off;

        }

        curbits =  ff_aac_spectral_bits[cb-1][curidx];

        vec     = &ff_aac_codebook_vectors[cb-1][curidx*dim];

        if (BT_UNSIGNED) {

            for (k = 0; k < dim; k++) {

                float t = fabsf(in[i+k]);

                float di;

                if (BT_ESC && vec[k] == 64.0f) { //FIXME: slow

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

        cost    += rd * lambda + curbits;

        resbits += curbits;

        if (cost >= uplim)

            return uplim;

        if (pb) {

            put_bits(pb, ff_aac_spectral_bits[cb-1][curidx], ff_aac_spectral_codes[cb-1][curidx]);

            if (BT_UNSIGNED)

                for (j = 0; j < dim; j++)

                    if (ff_aac_codebook_vectors[cb-1][curidx*dim+j] != 0.0f)

                        put_bits(pb, 1, in[i+j] < 0.0f);

            if (BT_ESC) {

                for (j = 0; j < 2; j++) {

                    if (ff_aac_codebook_vectors[cb-1][curidx*2+j] == 64.0f) {

                        int coef = av_clip(quant(fabsf(in[i+j]), Q), 0, 8191);

                        int len = av_log2(coef);



                        put_bits(pb, len - 4 + 1, (1 << (len - 4 + 1)) - 2);

                        put_bits(pb, len, coef & ((1 << len) - 1));

                    }

                }

            }

        }

    }



    if (bits)

        *bits = resbits;

    return cost;

}
