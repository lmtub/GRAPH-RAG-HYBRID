static int dvvideo_init(AVCodecContext *avctx)

{

    DVVideoContext *s = avctx->priv_data;

    DSPContext dsp;

    static int done=0;

    int i, j;



    if (!done) {

        VLC dv_vlc;

        uint16_t new_dv_vlc_bits[NB_DV_VLC*2];

        uint8_t new_dv_vlc_len[NB_DV_VLC*2];

        uint8_t new_dv_vlc_run[NB_DV_VLC*2];

        int16_t new_dv_vlc_level[NB_DV_VLC*2];



        done = 1;



        dv_vlc_map = av_mallocz_static(DV_VLC_MAP_LEV_SIZE*DV_VLC_MAP_RUN_SIZE*sizeof(struct dv_vlc_pair));

        if (!dv_vlc_map)

            return -ENOMEM;



        /* dv_anchor lets each thread know its Id */

        dv_anchor = av_malloc(12*27*sizeof(void*));

        if (!dv_anchor) {

            return -ENOMEM;

        }

        for (i=0; i<12*27; i++)

            dv_anchor[i] = (void*)(size_t)i;



        /* it's faster to include sign bit in a generic VLC parsing scheme */

        for (i=0, j=0; i<NB_DV_VLC; i++, j++) {

            new_dv_vlc_bits[j] = dv_vlc_bits[i];

            new_dv_vlc_len[j] = dv_vlc_len[i];

            new_dv_vlc_run[j] = dv_vlc_run[i];

            new_dv_vlc_level[j] = dv_vlc_level[i];



            if (dv_vlc_level[i]) {

                new_dv_vlc_bits[j] <<= 1;

                new_dv_vlc_len[j]++;



                j++;

                new_dv_vlc_bits[j] = (dv_vlc_bits[i] << 1) | 1;

                new_dv_vlc_len[j] = dv_vlc_len[i] + 1;

                new_dv_vlc_run[j] = dv_vlc_run[i];

                new_dv_vlc_level[j] = -dv_vlc_level[i];

            }

        }



        /* NOTE: as a trick, we use the fact the no codes are unused

           to accelerate the parsing of partial codes */

        init_vlc(&dv_vlc, TEX_VLC_BITS, j,

                 new_dv_vlc_len, 1, 1, new_dv_vlc_bits, 2, 2, 0);



        dv_rl_vlc = av_malloc(dv_vlc.table_size * sizeof(RL_VLC_ELEM));

        if (!dv_rl_vlc) {

            av_free(dv_anchor);

            return -ENOMEM;

        }

        for(i = 0; i < dv_vlc.table_size; i++){

            int code= dv_vlc.table[i][0];

            int len = dv_vlc.table[i][1];

            int level, run;



            if(len<0){ //more bits needed

                run= 0;

                level= code;

            } else {

                run=   new_dv_vlc_run[code] + 1;

                level= new_dv_vlc_level[code];

            }

            dv_rl_vlc[i].len = len;

            dv_rl_vlc[i].level = level;

            dv_rl_vlc[i].run = run;

        }

        free_vlc(&dv_vlc);



        for (i = 0; i < NB_DV_VLC - 1; i++) {

           if (dv_vlc_run[i] >= DV_VLC_MAP_RUN_SIZE)

               continue;

#ifdef DV_CODEC_TINY_TARGET

           if (dv_vlc_level[i] >= DV_VLC_MAP_LEV_SIZE)

               continue;

#endif



           if (dv_vlc_map[dv_vlc_run[i]][dv_vlc_level[i]].size != 0)

               continue;



           dv_vlc_map[dv_vlc_run[i]][dv_vlc_level[i]].vlc = dv_vlc_bits[i] <<

                                                            (!!dv_vlc_level[i]);

           dv_vlc_map[dv_vlc_run[i]][dv_vlc_level[i]].size = dv_vlc_len[i] +

                                                             (!!dv_vlc_level[i]);

        }

        for (i = 0; i < DV_VLC_MAP_RUN_SIZE; i++) {

#ifdef DV_CODEC_TINY_TARGET

           for (j = 1; j < DV_VLC_MAP_LEV_SIZE; j++) {

              if (dv_vlc_map[i][j].size == 0) {

                  dv_vlc_map[i][j].vlc = dv_vlc_map[0][j].vlc |

                            (dv_vlc_map[i-1][0].vlc << (dv_vlc_map[0][j].size));

                  dv_vlc_map[i][j].size = dv_vlc_map[i-1][0].size +

                                          dv_vlc_map[0][j].size;

              }

           }

#else

           for (j = 1; j < DV_VLC_MAP_LEV_SIZE/2; j++) {

              if (dv_vlc_map[i][j].size == 0) {

                  dv_vlc_map[i][j].vlc = dv_vlc_map[0][j].vlc |

                            (dv_vlc_map[i-1][0].vlc << (dv_vlc_map[0][j].size));

                  dv_vlc_map[i][j].size = dv_vlc_map[i-1][0].size +

                                          dv_vlc_map[0][j].size;

              }

              dv_vlc_map[i][((uint16_t)(-j))&0x1ff].vlc =

                                            dv_vlc_map[i][j].vlc | 1;

              dv_vlc_map[i][((uint16_t)(-j))&0x1ff].size =

                                            dv_vlc_map[i][j].size;

           }

#endif

        }

    }



    /* Generic DSP setup */

    dsputil_init(&dsp, avctx);

    s->get_pixels = dsp.get_pixels;



    /* 88DCT setup */

    s->fdct[0] = dsp.fdct;

    s->idct_put[0] = dsp.idct_put;

    for (i=0; i<64; i++)

       s->dv_zigzag[0][i] = dsp.idct_permutation[ff_zigzag_direct[i]];



    /* 248DCT setup */

    s->fdct[1] = dsp.fdct248;

    s->idct_put[1] = simple_idct248_put;  // FIXME: need to add it to DSP

    if(avctx->lowres){

        for (i=0; i<64; i++){

            int j= ff_zigzag248_direct[i];

            s->dv_zigzag[1][i] = dsp.idct_permutation[(j&7) + (j&8)*4 + (j&48)/2];

        }

    }else

        memcpy(s->dv_zigzag[1], ff_zigzag248_direct, 64);



    /* XXX: do it only for constant case */

    dv_build_unquantize_tables(s, dsp.idct_permutation);



    /* FIXME: I really don't think this should be here */

    if (dv_codec_profile(avctx))

        avctx->pix_fmt = dv_codec_profile(avctx)->pix_fmt;

    avctx->coded_frame = &s->picture;

    s->avctx= avctx;



    return 0;

}
