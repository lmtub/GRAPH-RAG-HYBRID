static av_cold int decode_init(AVCodecContext * avctx)

{

    MPADecodeContext *s = avctx->priv_data;

    static int init=0;

    int i, j, k;



    s->avctx = avctx;



    ff_mpadsp_init(&s->mpadsp);



    avctx->sample_fmt= OUT_FMT;

    s->error_recognition= avctx->error_recognition;



    if (!init && !avctx->parse_only) {

        int offset;



        /* scale factors table for layer 1/2 */

        for(i=0;i<64;i++) {

            int shift, mod;

            /* 1.0 (i = 3) is normalized to 2 ^ FRAC_BITS */

            shift = (i / 3);

            mod = i % 3;

            scale_factor_modshift[i] = mod | (shift << 2);

        }



        /* scale factor multiply for layer 1 */

        for(i=0;i<15;i++) {

            int n, norm;

            n = i + 2;

            norm = ((INT64_C(1) << n) * FRAC_ONE) / ((1 << n) - 1);

            scale_factor_mult[i][0] = MULLx(norm, FIXR(1.0          * 2.0), FRAC_BITS);

            scale_factor_mult[i][1] = MULLx(norm, FIXR(0.7937005259 * 2.0), FRAC_BITS);

            scale_factor_mult[i][2] = MULLx(norm, FIXR(0.6299605249 * 2.0), FRAC_BITS);

            av_dlog(avctx, "%d: norm=%x s=%x %x %x\n",

                    i, norm,

                    scale_factor_mult[i][0],

                    scale_factor_mult[i][1],

                    scale_factor_mult[i][2]);

        }



        RENAME(ff_mpa_synth_init)(RENAME(ff_mpa_synth_window));



        /* huffman decode tables */

        offset = 0;

        for(i=1;i<16;i++) {

            const HuffTable *h = &mpa_huff_tables[i];

            int xsize, x, y;

            uint8_t  tmp_bits [512];

            uint16_t tmp_codes[512];



            memset(tmp_bits , 0, sizeof(tmp_bits ));

            memset(tmp_codes, 0, sizeof(tmp_codes));



            xsize = h->xsize;



            j = 0;

            for(x=0;x<xsize;x++) {

                for(y=0;y<xsize;y++){

                    tmp_bits [(x << 5) | y | ((x&&y)<<4)]= h->bits [j  ];

                    tmp_codes[(x << 5) | y | ((x&&y)<<4)]= h->codes[j++];

                }

            }



            /* XXX: fail test */

            huff_vlc[i].table = huff_vlc_tables+offset;

            huff_vlc[i].table_allocated = huff_vlc_tables_sizes[i];

            init_vlc(&huff_vlc[i], 7, 512,

                     tmp_bits, 1, 1, tmp_codes, 2, 2,

                     INIT_VLC_USE_NEW_STATIC);

            offset += huff_vlc_tables_sizes[i];

        }

        assert(offset == FF_ARRAY_ELEMS(huff_vlc_tables));



        offset = 0;

        for(i=0;i<2;i++) {

            huff_quad_vlc[i].table = huff_quad_vlc_tables+offset;

            huff_quad_vlc[i].table_allocated = huff_quad_vlc_tables_sizes[i];

            init_vlc(&huff_quad_vlc[i], i == 0 ? 7 : 4, 16,

                     mpa_quad_bits[i], 1, 1, mpa_quad_codes[i], 1, 1,

                     INIT_VLC_USE_NEW_STATIC);

            offset += huff_quad_vlc_tables_sizes[i];

        }

        assert(offset == FF_ARRAY_ELEMS(huff_quad_vlc_tables));



        for(i=0;i<9;i++) {

            k = 0;

            for(j=0;j<22;j++) {

                band_index_long[i][j] = k;

                k += band_size_long[i][j];

            }

            band_index_long[i][22] = k;

        }



        /* compute n ^ (4/3) and store it in mantissa/exp format */



        int_pow_init();

        mpegaudio_tableinit();



        for (i = 0; i < 4; i++)

            if (ff_mpa_quant_bits[i] < 0)

                for (j = 0; j < (1<<(-ff_mpa_quant_bits[i]+1)); j++) {

                    int val1, val2, val3, steps;

                    int val = j;

                    steps  = ff_mpa_quant_steps[i];

                    val1 = val % steps;

                    val /= steps;

                    val2 = val % steps;

                    val3 = val / steps;

                    division_tabs[i][j] = val1 + (val2 << 4) + (val3 << 8);

                }





        for(i=0;i<7;i++) {

            float f;

            INTFLOAT v;

            if (i != 6) {

                f = tan((double)i * M_PI / 12.0);

                v = FIXR(f / (1.0 + f));

            } else {

                v = FIXR(1.0);

            }

            is_table[0][i] = v;

            is_table[1][6 - i] = v;

        }

        /* invalid values */

        for(i=7;i<16;i++)

            is_table[0][i] = is_table[1][i] = 0.0;



        for(i=0;i<16;i++) {

            double f;

            int e, k;



            for(j=0;j<2;j++) {

                e = -(j + 1) * ((i + 1) >> 1);

                f = pow(2.0, e / 4.0);

                k = i & 1;

                is_table_lsf[j][k ^ 1][i] = FIXR(f);

                is_table_lsf[j][k][i] = FIXR(1.0);

                av_dlog(avctx, "is_table_lsf %d %d: %x %x\n",

                        i, j, is_table_lsf[j][0][i], is_table_lsf[j][1][i]);

            }

        }



        for(i=0;i<8;i++) {

            float ci, cs, ca;

            ci = ci_table[i];

            cs = 1.0 / sqrt(1.0 + ci * ci);

            ca = cs * ci;

            csa_table[i][0] = FIXHR(cs/4);

            csa_table[i][1] = FIXHR(ca/4);

            csa_table[i][2] = FIXHR(ca/4) + FIXHR(cs/4);

            csa_table[i][3] = FIXHR(ca/4) - FIXHR(cs/4);

            csa_table_float[i][0] = cs;

            csa_table_float[i][1] = ca;

            csa_table_float[i][2] = ca + cs;

            csa_table_float[i][3] = ca - cs;

        }



        /* compute mdct windows */

        for(i=0;i<36;i++) {

            for(j=0; j<4; j++){

                double d;



                if(j==2 && i%3 != 1)

                    continue;



                d= sin(M_PI * (i + 0.5) / 36.0);

                if(j==1){

                    if     (i>=30) d= 0;

                    else if(i>=24) d= sin(M_PI * (i - 18 + 0.5) / 12.0);

                    else if(i>=18) d= 1;

                }else if(j==3){

                    if     (i<  6) d= 0;

                    else if(i< 12) d= sin(M_PI * (i -  6 + 0.5) / 12.0);

                    else if(i< 18) d= 1;

                }

                //merge last stage of imdct into the window coefficients

                d*= 0.5 / cos(M_PI*(2*i + 19)/72);



                if(j==2)

                    mdct_win[j][i/3] = FIXHR((d / (1<<5)));

                else

                    mdct_win[j][i  ] = FIXHR((d / (1<<5)));

            }

        }



        /* NOTE: we do frequency inversion adter the MDCT by changing

           the sign of the right window coefs */

        for(j=0;j<4;j++) {

            for(i=0;i<36;i+=2) {

                mdct_win[j + 4][i] = mdct_win[j][i];

                mdct_win[j + 4][i + 1] = -mdct_win[j][i + 1];

            }

        }



        init = 1;

    }



    if (avctx->codec_id == CODEC_ID_MP3ADU)

        s->adu_mode = 1;

    return 0;

}
