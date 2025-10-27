static void decode_block(BinkAudioContext *s, short *out, int use_dct)

{

    int ch, i, j, k;

    float q, quant[25];

    int width, coeff;

    GetBitContext *gb = &s->gb;



    if (use_dct)

        skip_bits(gb, 2);



    for (ch = 0; ch < s->channels; ch++) {

        FFTSample *coeffs = s->coeffs_ptr[ch];

        q = 0.0f;

        coeffs[0] = get_float(gb) * s->root;

        coeffs[1] = get_float(gb) * s->root;



        for (i = 0; i < s->num_bands; i++) {

            /* constant is result of 0.066399999/log10(M_E) */

            int value = get_bits(gb, 8);

            quant[i] = expf(FFMIN(value, 95) * 0.15289164787221953823f) * s->root;

        }



        // find band (k)

        for (k = 0; s->bands[k] < 1; k++) {

            q = quant[k];

        }



        // parse coefficients

        i = 2;

        while (i < s->frame_len) {

            if (get_bits1(gb)) {

                j = i + rle_length_tab[get_bits(gb, 4)] * 8;

            } else {

                j = i + 8;

            }



            j = FFMIN(j, s->frame_len);



            width = get_bits(gb, 4);

            if (width == 0) {

                memset(coeffs + i, 0, (j - i) * sizeof(*coeffs));

                i = j;

                while (s->bands[k] < i)

                    q = quant[k++];

            } else {

                while (i < j) {

                    if (s->bands[k] == i)

                        q = quant[k++];

                    coeff = get_bits(gb, width);

                    if (coeff) {

                        if (get_bits1(gb))

                            coeffs[i] = -q * coeff;

                        else

                            coeffs[i] =  q * coeff;

                    } else {

                        coeffs[i] = 0.0f;

                    }

                    i++;

                }

            }

        }



        if (CONFIG_BINKAUDIO_DCT_DECODER && use_dct) {

            coeffs[0] /= 0.5;

            ff_dct_calc (&s->trans.dct,  coeffs);

            s->dsp.vector_fmul_scalar(coeffs, coeffs, s->frame_len / 2, s->frame_len);

        }

        else if (CONFIG_BINKAUDIO_RDFT_DECODER)

            ff_rdft_calc(&s->trans.rdft, coeffs);

    }



    s->fmt_conv.float_to_int16_interleave(out, (const float **)s->coeffs_ptr,

                                          s->frame_len, s->channels);



    if (!s->first) {

        int count = s->overlap_len * s->channels;

        int shift = av_log2(count);

        for (i = 0; i < count; i++) {

            out[i] = (s->previous[i] * (count - i) + out[i] * i) >> shift;

        }

    }



    memcpy(s->previous, out + s->block_size,

           s->overlap_len * s->channels * sizeof(*out));



    s->first = 0;

}
