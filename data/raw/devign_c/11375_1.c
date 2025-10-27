static void encode_frame(MpegAudioContext *s,

                         unsigned char bit_alloc[MPA_MAX_CHANNELS][SBLIMIT],

                         int padding)

{

    int i, j, k, l, bit_alloc_bits, b, ch;

    unsigned char *sf;

    int q[3];

    PutBitContext *p = &s->pb;



    /* header */



    put_bits(p, 12, 0xfff);

    put_bits(p, 1, 1 - s->lsf); /* 1 = mpeg1 ID, 0 = mpeg2 lsf ID */

    put_bits(p, 2, 4-2);  /* layer 2 */

    put_bits(p, 1, 1); /* no error protection */

    put_bits(p, 4, s->bitrate_index);

    put_bits(p, 2, s->freq_index);

    put_bits(p, 1, s->do_padding); /* use padding */

    put_bits(p, 1, 0);             /* private_bit */

    put_bits(p, 2, s->nb_channels == 2 ? MPA_STEREO : MPA_MONO);

    put_bits(p, 2, 0); /* mode_ext */

    put_bits(p, 1, 0); /* no copyright */

    put_bits(p, 1, 1); /* original */

    put_bits(p, 2, 0); /* no emphasis */



    /* bit allocation */

    j = 0;

    for(i=0;i<s->sblimit;i++) {

        bit_alloc_bits = s->alloc_table[j];

        for(ch=0;ch<s->nb_channels;ch++) {

            put_bits(p, bit_alloc_bits, bit_alloc[ch][i]);

        }

        j += 1 << bit_alloc_bits;

    }



    /* scale codes */

    for(i=0;i<s->sblimit;i++) {

        for(ch=0;ch<s->nb_channels;ch++) {

            if (bit_alloc[ch][i])

                put_bits(p, 2, s->scale_code[ch][i]);

        }

    }



    /* scale factors */

    for(i=0;i<s->sblimit;i++) {

        for(ch=0;ch<s->nb_channels;ch++) {

            if (bit_alloc[ch][i]) {

                sf = &s->scale_factors[ch][i][0];

                switch(s->scale_code[ch][i]) {

                case 0:

                    put_bits(p, 6, sf[0]);

                    put_bits(p, 6, sf[1]);

                    put_bits(p, 6, sf[2]);

                    break;

                case 3:

                case 1:

                    put_bits(p, 6, sf[0]);

                    put_bits(p, 6, sf[2]);

                    break;

                case 2:

                    put_bits(p, 6, sf[0]);

                    break;

                }

            }

        }

    }



    /* quantization & write sub band samples */



    for(k=0;k<3;k++) {

        for(l=0;l<12;l+=3) {

            j = 0;

            for(i=0;i<s->sblimit;i++) {

                bit_alloc_bits = s->alloc_table[j];

                for(ch=0;ch<s->nb_channels;ch++) {

                    b = bit_alloc[ch][i];

                    if (b) {

                        int qindex, steps, m, sample, bits;

                        /* we encode 3 sub band samples of the same sub band at a time */

                        qindex = s->alloc_table[j+b];

                        steps = ff_mpa_quant_steps[qindex];

                        for(m=0;m<3;m++) {

                            sample = s->sb_samples[ch][k][l + m][i];

                            /* divide by scale factor */

#if USE_FLOATS

                            {

                                float a;

                                a = (float)sample * s->scale_factor_inv_table[s->scale_factors[ch][i][k]];

                                q[m] = (int)((a + 1.0) * steps * 0.5);

                            }

#else

                            {

                                int q1, e, shift, mult;

                                e = s->scale_factors[ch][i][k];

                                shift = s->scale_factor_shift[e];

                                mult = s->scale_factor_mult[e];



                                /* normalize to P bits */

                                if (shift < 0)

                                    q1 = sample << (-shift);

                                else

                                    q1 = sample >> shift;

                                q1 = (q1 * mult) >> P;

                                q1 += 1 << P;

                                if (q1 < 0)

                                    q1 = 0;

                                q[m] = (unsigned)(q1 * steps) >> (P + 1);

                            }

#endif

                            if (q[m] >= steps)

                                q[m] = steps - 1;

                            av_assert2(q[m] >= 0 && q[m] < steps);

                        }

                        bits = ff_mpa_quant_bits[qindex];

                        if (bits < 0) {

                            /* group the 3 values to save bits */

                            put_bits(p, -bits,

                                     q[0] + steps * (q[1] + steps * q[2]));

                        } else {

                            put_bits(p, bits, q[0]);

                            put_bits(p, bits, q[1]);

                            put_bits(p, bits, q[2]);

                        }

                    }

                }

                /* next subband in alloc table */

                j += 1 << bit_alloc_bits;

            }

        }

    }



    /* padding */

    for(i=0;i<padding;i++)

        put_bits(p, 1, 0);



    /* flush */

    flush_put_bits(p);

}
