av_cold int ff_dvvideo_init(AVCodecContext *avctx)

{

    DVVideoContext *s = avctx->priv_data;

    DSPContext dsp;

    static int done = 0;

    int i, j;



    if (!done) {

        VLC dv_vlc;

        uint16_t new_dv_vlc_bits[NB_DV_VLC*2];

        uint8_t  new_dv_vlc_len[NB_DV_VLC*2];

        uint8_t  new_dv_vlc_run[NB_DV_VLC*2];

        int16_t  new_dv_vlc_level[NB_DV_VLC*2];



        done = 1;



        /* it's faster to include sign bit in a generic VLC parsing scheme */

        for (i = 0, j = 0; i < NB_DV_VLC; i++, j++) {

            new_dv_vlc_bits[j]  = dv_vlc_bits[i];

            new_dv_vlc_len[j]   = dv_vlc_len[i];

            new_dv_vlc_run[j]   = dv_vlc_run[i];

            new_dv_vlc_level[j] = dv_vlc_level[i];



            if (dv_vlc_level[i]) {

                new_dv_vlc_bits[j] <<= 1;

                new_dv_vlc_len[j]++;



                j++;

                new_dv_vlc_bits[j]  = (dv_vlc_bits[i] << 1) | 1;

                new_dv_vlc_len[j]   =  dv_vlc_len[i] + 1;

                new_dv_vlc_run[j]   =  dv_vlc_run[i];

                new_dv_vlc_level[j] = -dv_vlc_level[i];

            }

        }



        /* NOTE: as a trick, we use the fact the no codes are unused

           to accelerate the parsing of partial codes */

        init_vlc(&dv_vlc, TEX_VLC_BITS, j,

                 new_dv_vlc_len, 1, 1, new_dv_vlc_bits, 2, 2, 0);

        assert(dv_vlc.table_size == 1184);



        for (i = 0; i < dv_vlc.table_size; i++){

            int code = dv_vlc.table[i][0];

            int len  = dv_vlc.table[i][1];

            int level, run;



            if (len < 0){ //more bits needed

                run   = 0;

                level = code;

            } else {

                run   = new_dv_vlc_run  [code] + 1;

                level = new_dv_vlc_level[code];

            }

            ff_dv_rl_vlc[i].len   = len;

            ff_dv_rl_vlc[i].level = level;

            ff_dv_rl_vlc[i].run   = run;

        }

        ff_free_vlc(&dv_vlc);

    }



    /* Generic DSP setup */


    ff_dsputil_init(&dsp, avctx);

    ff_set_cmp(&dsp, dsp.ildct_cmp, avctx->ildct_cmp);

    s->get_pixels = dsp.get_pixels;

    s->ildct_cmp = dsp.ildct_cmp[5];



    /* 88DCT setup */

    s->fdct[0]     = dsp.fdct;

    s->idct_put[0] = dsp.idct_put;

    for (i = 0; i < 64; i++)

       s->dv_zigzag[0][i] = dsp.idct_permutation[ff_zigzag_direct[i]];



    /* 248DCT setup */

    s->fdct[1]     = dsp.fdct248;

    s->idct_put[1] = ff_simple_idct248_put;  // FIXME: need to add it to DSP

    if (avctx->lowres){

        for (i = 0; i < 64; i++){

            int j = ff_zigzag248_direct[i];

            s->dv_zigzag[1][i] = dsp.idct_permutation[(j & 7) + (j & 8) * 4 + (j & 48) / 2];

        }

    }else

        memcpy(s->dv_zigzag[1], ff_zigzag248_direct, 64);



    avctx->coded_frame = &s->picture;

    s->avctx = avctx;

    avctx->chroma_sample_location = AVCHROMA_LOC_TOPLEFT;



    return 0;

}