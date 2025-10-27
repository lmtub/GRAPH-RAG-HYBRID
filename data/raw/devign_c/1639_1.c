static int wma_decode_init(AVCodecContext * avctx)

{

    WMADecodeContext *s = avctx->priv_data;

    int i, flags1, flags2;

    float *window;

    uint8_t *extradata;

    float bps1, high_freq;

    volatile float bps;

    int sample_rate1;

    int coef_vlc_table;

    

    s->sample_rate = avctx->sample_rate;

    s->nb_channels = avctx->channels;

    s->bit_rate = avctx->bit_rate;

    s->block_align = avctx->block_align;



    if (avctx->codec->id == CODEC_ID_WMAV1) {

        s->version = 1;

    } else {

        s->version = 2;

    }

    

    /* extract flag infos */

    flags1 = 0;

    flags2 = 0;

    extradata = avctx->extradata;

    if (s->version == 1 && avctx->extradata_size >= 4) {

        flags1 = extradata[0] | (extradata[1] << 8);

        flags2 = extradata[2] | (extradata[3] << 8);

    } else if (s->version == 2 && avctx->extradata_size >= 6) {

        flags1 = extradata[0] | (extradata[1] << 8) | 

            (extradata[2] << 16) | (extradata[3] << 24);

        flags2 = extradata[4] | (extradata[5] << 8);

    }

    s->use_exp_vlc = flags2 & 0x0001;

    s->use_bit_reservoir = flags2 & 0x0002;

    s->use_variable_block_len = flags2 & 0x0004;



    /* compute MDCT block size */

    if (s->sample_rate <= 16000) {

        s->frame_len_bits = 9;

    } else if (s->sample_rate <= 22050 || 

               (s->sample_rate <= 32000 && s->version == 1)) {

        s->frame_len_bits = 10;

    } else {

        s->frame_len_bits = 11;

    }

    s->frame_len = 1 << s->frame_len_bits;

    if (s->use_variable_block_len) {

        int nb_max, nb;

        nb = ((flags2 >> 3) & 3) + 1;

        if ((s->bit_rate / s->nb_channels) >= 32000)

            nb += 2;

        nb_max = s->frame_len_bits - BLOCK_MIN_BITS;

        if (nb > nb_max)

            nb = nb_max;

        s->nb_block_sizes = nb + 1;

    } else {

        s->nb_block_sizes = 1;

    }



    /* init rate dependant parameters */

    s->use_noise_coding = 1;

    high_freq = s->sample_rate * 0.5;



    /* if version 2, then the rates are normalized */

    sample_rate1 = s->sample_rate;

    if (s->version == 2) {

        if (sample_rate1 >= 44100) 

            sample_rate1 = 44100;

        else if (sample_rate1 >= 22050) 

            sample_rate1 = 22050;

        else if (sample_rate1 >= 16000) 

            sample_rate1 = 16000;

        else if (sample_rate1 >= 11025) 

            sample_rate1 = 11025;

        else if (sample_rate1 >= 8000) 

            sample_rate1 = 8000;

    }



    bps = (float)s->bit_rate / (float)(s->nb_channels * s->sample_rate);

    s->byte_offset_bits = av_log2((int)(bps * s->frame_len / 8.0)) + 2;



    /* compute high frequency value and choose if noise coding should

       be activated */

    bps1 = bps;

    if (s->nb_channels == 2)

        bps1 = bps * 1.6;

    if (sample_rate1 == 44100) {

        if (bps1 >= 0.61)

            s->use_noise_coding = 0;

        else

            high_freq = high_freq * 0.4;

    } else if (sample_rate1 == 22050) {

        if (bps1 >= 1.16)

            s->use_noise_coding = 0;

        else if (bps1 >= 0.72) 

            high_freq = high_freq * 0.7;

        else

            high_freq = high_freq * 0.6;

    } else if (sample_rate1 == 16000) {

        if (bps > 0.5)

            high_freq = high_freq * 0.5;

        else

            high_freq = high_freq * 0.3;

    } else if (sample_rate1 == 11025) {

        high_freq = high_freq * 0.7;

    } else if (sample_rate1 == 8000) {

        if (bps <= 0.625) {

            high_freq = high_freq * 0.5;

        } else if (bps > 0.75) {

            s->use_noise_coding = 0;

        } else {

            high_freq = high_freq * 0.65;

        }

    } else {

        if (bps >= 0.8) {

            high_freq = high_freq * 0.75;

        } else if (bps >= 0.6) {

            high_freq = high_freq * 0.6;

        } else {

            high_freq = high_freq * 0.5;

        }

    }

    dprintf("flags1=0x%x flags2=0x%x\n", flags1, flags2);

    dprintf("version=%d channels=%d sample_rate=%d bitrate=%d block_align=%d\n",

           s->version, s->nb_channels, s->sample_rate, s->bit_rate, 

           s->block_align);

    dprintf("bps=%f bps1=%f high_freq=%f bitoffset=%d\n", 

           bps, bps1, high_freq, s->byte_offset_bits);

    dprintf("use_noise_coding=%d use_exp_vlc=%d nb_block_sizes=%d\n",

           s->use_noise_coding, s->use_exp_vlc, s->nb_block_sizes);



    /* compute the scale factor band sizes for each MDCT block size */

    {

        int a, b, pos, lpos, k, block_len, i, j, n;

        const uint8_t *table;

        

        if (s->version == 1) {

            s->coefs_start = 3;

        } else {

            s->coefs_start = 0;

        }

        for(k = 0; k < s->nb_block_sizes; k++) {

            block_len = s->frame_len >> k;



            if (s->version == 1) {

                lpos = 0;

                for(i=0;i<25;i++) {

                    a = wma_critical_freqs[i];

                    b = s->sample_rate;

                    pos = ((block_len * 2 * a)  + (b >> 1)) / b;

                    if (pos > block_len) 

                        pos = block_len;

                    s->exponent_bands[0][i] = pos - lpos;

                    if (pos >= block_len) {

                        i++;

                        break;

                    }

                    lpos = pos;

                }

                s->exponent_sizes[0] = i;

            } else {

                /* hardcoded tables */

                table = NULL;

                a = s->frame_len_bits - BLOCK_MIN_BITS - k;

                if (a < 3) {

                    if (s->sample_rate >= 44100)

                        table = exponent_band_44100[a];

                    else if (s->sample_rate >= 32000)

                        table = exponent_band_32000[a];

                    else if (s->sample_rate >= 22050)

                        table = exponent_band_22050[a];

                }

                if (table) {

                    n = *table++;

                    for(i=0;i<n;i++)

                        s->exponent_bands[k][i] = table[i];

                    s->exponent_sizes[k] = n;

                } else {

                    j = 0;

                    lpos = 0;

                    for(i=0;i<25;i++) {

                        a = wma_critical_freqs[i];

                        b = s->sample_rate;

                        pos = ((block_len * 2 * a)  + (b << 1)) / (4 * b);

                        pos <<= 2;

                        if (pos > block_len) 

                            pos = block_len;

                        if (pos > lpos)

                            s->exponent_bands[k][j++] = pos - lpos;

                        if (pos >= block_len)

                            break;

                        lpos = pos;

                    }

                    s->exponent_sizes[k] = j;

                }

            }



            /* max number of coefs */

            s->coefs_end[k] = (s->frame_len - ((s->frame_len * 9) / 100)) >> k;

            /* high freq computation */

            s->high_band_start[k] = (int)((block_len * 2 * high_freq) / 

                                          s->sample_rate + 0.5);

            n = s->exponent_sizes[k];

            j = 0;

            pos = 0;

            for(i=0;i<n;i++) {

                int start, end;

                start = pos;

                pos += s->exponent_bands[k][i];

                end = pos;

                if (start < s->high_band_start[k])

                    start = s->high_band_start[k];

                if (end > s->coefs_end[k])

                    end = s->coefs_end[k];

                if (end > start)

                    s->exponent_high_bands[k][j++] = end - start;

            }

            s->exponent_high_sizes[k] = j;

#if 0

            tprintf("%5d: coefs_end=%d high_band_start=%d nb_high_bands=%d: ",

                  s->frame_len >> k, 

                  s->coefs_end[k],

                  s->high_band_start[k],

                  s->exponent_high_sizes[k]);

            for(j=0;j<s->exponent_high_sizes[k];j++)

                tprintf(" %d", s->exponent_high_bands[k][j]);

            tprintf("\n");

#endif

        }

    }



#ifdef TRACE

    {

        int i, j;

        for(i = 0; i < s->nb_block_sizes; i++) {

            tprintf("%5d: n=%2d:", 

                   s->frame_len >> i, 

                   s->exponent_sizes[i]);

            for(j=0;j<s->exponent_sizes[i];j++)

                tprintf(" %d", s->exponent_bands[i][j]);

            tprintf("\n");

        }

    }

#endif



    /* init MDCT */

    for(i = 0; i < s->nb_block_sizes; i++)

        ff_mdct_init(&s->mdct_ctx[i], s->frame_len_bits - i + 1, 1);

    

    /* init MDCT windows : simple sinus window */

    for(i = 0; i < s->nb_block_sizes; i++) {

        int n, j;

        float alpha;

        n = 1 << (s->frame_len_bits - i);

        window = av_malloc(sizeof(float) * n);

        alpha = M_PI / (2.0 * n);

        for(j=0;j<n;j++) {

            window[n - j - 1] = sin((j + 0.5) * alpha);

        }

        s->windows[i] = window;

    }



    s->reset_block_lengths = 1;

    

    if (s->use_noise_coding) {



        /* init the noise generator */

        if (s->use_exp_vlc)

            s->noise_mult = 0.02;

        else

            s->noise_mult = 0.04;

               

#ifdef TRACE

        for(i=0;i<NOISE_TAB_SIZE;i++)

            s->noise_table[i] = 1.0 * s->noise_mult;

#else

        {

            unsigned int seed;

            float norm;

            seed = 1;

            norm = (1.0 / (float)(1LL << 31)) * sqrt(3) * s->noise_mult;

            for(i=0;i<NOISE_TAB_SIZE;i++) {

                seed = seed * 314159 + 1;

                s->noise_table[i] = (float)((int)seed) * norm;

            }

        }

#endif

        init_vlc(&s->hgain_vlc, 9, sizeof(hgain_huffbits), 

                 hgain_huffbits, 1, 1,

                 hgain_huffcodes, 2, 2);

    }



    if (s->use_exp_vlc) {

        init_vlc(&s->exp_vlc, 9, sizeof(scale_huffbits), 

                 scale_huffbits, 1, 1,

                 scale_huffcodes, 4, 4);

    } else {

        wma_lsp_to_curve_init(s, s->frame_len);

    }



    /* choose the VLC tables for the coefficients */

    coef_vlc_table = 2;

    if (s->sample_rate >= 32000) {

        if (bps1 < 0.72)

            coef_vlc_table = 0;

        else if (bps1 < 1.16)

            coef_vlc_table = 1;

    }



    init_coef_vlc(&s->coef_vlc[0], &s->run_table[0], &s->level_table[0],

                  &coef_vlcs[coef_vlc_table * 2]);

    init_coef_vlc(&s->coef_vlc[1], &s->run_table[1], &s->level_table[1],

                  &coef_vlcs[coef_vlc_table * 2 + 1]);

    return 0;

}
