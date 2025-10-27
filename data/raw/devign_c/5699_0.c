static av_cold int vp3_decode_init(AVCodecContext *avctx)

{

    Vp3DecodeContext *s = avctx->priv_data;

    int i, inter, plane;

    int c_width;

    int c_height;

    int y_fragment_count, c_fragment_count;



    if (avctx->codec_tag == MKTAG('V','P','3','0'))

        s->version = 0;

    else

        s->version = 1;



    s->avctx = avctx;

    s->width = FFALIGN(avctx->width, 16);

    s->height = FFALIGN(avctx->height, 16);

    if (avctx->pix_fmt == PIX_FMT_NONE)

        avctx->pix_fmt = PIX_FMT_YUV420P;

    avctx->chroma_sample_location = AVCHROMA_LOC_CENTER;

    if(avctx->idct_algo==FF_IDCT_AUTO)

        avctx->idct_algo=FF_IDCT_VP3;

    ff_dsputil_init(&s->dsp, avctx);



    ff_init_scantable(s->dsp.idct_permutation, &s->scantable, ff_zigzag_direct);



    /* initialize to an impossible value which will force a recalculation

     * in the first frame decode */

    for (i = 0; i < 3; i++)

        s->qps[i] = -1;



    avcodec_get_chroma_sub_sample(avctx->pix_fmt, &s->chroma_x_shift, &s->chroma_y_shift);



    s->y_superblock_width = (s->width + 31) / 32;

    s->y_superblock_height = (s->height + 31) / 32;

    s->y_superblock_count = s->y_superblock_width * s->y_superblock_height;



    /* work out the dimensions for the C planes */

    c_width = s->width >> s->chroma_x_shift;

    c_height = s->height >> s->chroma_y_shift;

    s->c_superblock_width = (c_width + 31) / 32;

    s->c_superblock_height = (c_height + 31) / 32;

    s->c_superblock_count = s->c_superblock_width * s->c_superblock_height;



    s->superblock_count = s->y_superblock_count + (s->c_superblock_count * 2);

    s->u_superblock_start = s->y_superblock_count;

    s->v_superblock_start = s->u_superblock_start + s->c_superblock_count;



    s->macroblock_width = (s->width + 15) / 16;

    s->macroblock_height = (s->height + 15) / 16;

    s->macroblock_count = s->macroblock_width * s->macroblock_height;



    s->fragment_width[0] = s->width / FRAGMENT_PIXELS;

    s->fragment_height[0] = s->height / FRAGMENT_PIXELS;

    s->fragment_width[1]  = s->fragment_width[0]  >> s->chroma_x_shift;

    s->fragment_height[1] = s->fragment_height[0] >> s->chroma_y_shift;



    /* fragment count covers all 8x8 blocks for all 3 planes */

    y_fragment_count     = s->fragment_width[0] * s->fragment_height[0];

    c_fragment_count     = s->fragment_width[1] * s->fragment_height[1];

    s->fragment_count    = y_fragment_count + 2*c_fragment_count;

    s->fragment_start[1] = y_fragment_count;

    s->fragment_start[2] = y_fragment_count + c_fragment_count;



    if (!s->theora_tables)

    {

        for (i = 0; i < 64; i++) {

            s->coded_dc_scale_factor[i] = vp31_dc_scale_factor[i];

            s->coded_ac_scale_factor[i] = vp31_ac_scale_factor[i];

            s->base_matrix[0][i] = vp31_intra_y_dequant[i];

            s->base_matrix[1][i] = vp31_intra_c_dequant[i];

            s->base_matrix[2][i] = vp31_inter_dequant[i];

            s->filter_limit_values[i] = vp31_filter_limit_values[i];

        }



        for(inter=0; inter<2; inter++){

            for(plane=0; plane<3; plane++){

                s->qr_count[inter][plane]= 1;

                s->qr_size [inter][plane][0]= 63;

                s->qr_base [inter][plane][0]=

                s->qr_base [inter][plane][1]= 2*inter + (!!plane)*!inter;

            }

        }



        /* init VLC tables */

        for (i = 0; i < 16; i++) {



            /* DC histograms */

            init_vlc(&s->dc_vlc[i], 11, 32,

                &dc_bias[i][0][1], 4, 2,

                &dc_bias[i][0][0], 4, 2, 0);



            /* group 1 AC histograms */

            init_vlc(&s->ac_vlc_1[i], 11, 32,

                &ac_bias_0[i][0][1], 4, 2,

                &ac_bias_0[i][0][0], 4, 2, 0);



            /* group 2 AC histograms */

            init_vlc(&s->ac_vlc_2[i], 11, 32,

                &ac_bias_1[i][0][1], 4, 2,

                &ac_bias_1[i][0][0], 4, 2, 0);



            /* group 3 AC histograms */

            init_vlc(&s->ac_vlc_3[i], 11, 32,

                &ac_bias_2[i][0][1], 4, 2,

                &ac_bias_2[i][0][0], 4, 2, 0);



            /* group 4 AC histograms */

            init_vlc(&s->ac_vlc_4[i], 11, 32,

                &ac_bias_3[i][0][1], 4, 2,

                &ac_bias_3[i][0][0], 4, 2, 0);

        }

    } else {



        for (i = 0; i < 16; i++) {

            /* DC histograms */

            if (init_vlc(&s->dc_vlc[i], 11, 32,

                &s->huffman_table[i][0][1], 8, 4,

                &s->huffman_table[i][0][0], 8, 4, 0) < 0)

                goto vlc_fail;



            /* group 1 AC histograms */

            if (init_vlc(&s->ac_vlc_1[i], 11, 32,

                &s->huffman_table[i+16][0][1], 8, 4,

                &s->huffman_table[i+16][0][0], 8, 4, 0) < 0)

                goto vlc_fail;



            /* group 2 AC histograms */

            if (init_vlc(&s->ac_vlc_2[i], 11, 32,

                &s->huffman_table[i+16*2][0][1], 8, 4,

                &s->huffman_table[i+16*2][0][0], 8, 4, 0) < 0)

                goto vlc_fail;



            /* group 3 AC histograms */

            if (init_vlc(&s->ac_vlc_3[i], 11, 32,

                &s->huffman_table[i+16*3][0][1], 8, 4,

                &s->huffman_table[i+16*3][0][0], 8, 4, 0) < 0)

                goto vlc_fail;



            /* group 4 AC histograms */

            if (init_vlc(&s->ac_vlc_4[i], 11, 32,

                &s->huffman_table[i+16*4][0][1], 8, 4,

                &s->huffman_table[i+16*4][0][0], 8, 4, 0) < 0)

                goto vlc_fail;

        }

    }



    init_vlc(&s->superblock_run_length_vlc, 6, 34,

        &superblock_run_length_vlc_table[0][1], 4, 2,

        &superblock_run_length_vlc_table[0][0], 4, 2, 0);



    init_vlc(&s->fragment_run_length_vlc, 5, 30,

        &fragment_run_length_vlc_table[0][1], 4, 2,

        &fragment_run_length_vlc_table[0][0], 4, 2, 0);



    init_vlc(&s->mode_code_vlc, 3, 8,

        &mode_code_vlc_table[0][1], 2, 1,

        &mode_code_vlc_table[0][0], 2, 1, 0);



    init_vlc(&s->motion_vector_vlc, 6, 63,

        &motion_vector_vlc_table[0][1], 2, 1,

        &motion_vector_vlc_table[0][0], 2, 1, 0);



    for (i = 0; i < 3; i++) {

        s->current_frame.data[i] = NULL;

        s->last_frame.data[i] = NULL;

        s->golden_frame.data[i] = NULL;

    }



    return allocate_tables(avctx);



vlc_fail:

    av_log(avctx, AV_LOG_FATAL, "Invalid huffman table\n");

    return -1;

}
