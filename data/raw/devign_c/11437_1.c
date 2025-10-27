static int vp3_decode_init(AVCodecContext *avctx)

{

    Vp3DecodeContext *s = avctx->priv_data;

    int i;



    s->avctx = avctx;

    s->width = avctx->width;

    s->height = avctx->height;

    avctx->pix_fmt = PIX_FMT_YUV420P;

    avctx->has_b_frames = 0;

    dsputil_init(&s->dsp, avctx);



    /* initialize to an impossible value which will force a recalculation

     * in the first frame decode */

    s->quality_index = -1;



    s->superblock_width = (s->width + 31) / 32;

    s->superblock_height = (s->height + 31) / 32;

    s->superblock_count = s->superblock_width * s->superblock_height * 3 / 2;

    s->u_superblock_start = s->superblock_width * s->superblock_height;

    s->v_superblock_start = s->superblock_width * s->superblock_height * 5 / 4;

    s->superblock_coding = av_malloc(s->superblock_count);



    s->macroblock_width = (s->width + 15) / 16;

    s->macroblock_height = (s->height + 15) / 16;

    s->macroblock_count = s->macroblock_width * s->macroblock_height;



    s->fragment_width = s->width / FRAGMENT_PIXELS;

    s->fragment_height = s->height / FRAGMENT_PIXELS;



    /* fragment count covers all 8x8 blocks for all 3 planes */

    s->fragment_count = s->fragment_width * s->fragment_height * 3 / 2;

    s->u_fragment_start = s->fragment_width * s->fragment_height;

    s->v_fragment_start = s->fragment_width * s->fragment_height * 5 / 4;



    debug_init("  width: %d x %d\n", s->width, s->height);

    debug_init("  superblocks: %d x %d, %d total\n",

        s->superblock_width, s->superblock_height, s->superblock_count);

    debug_init("  macroblocks: %d x %d, %d total\n",

        s->macroblock_width, s->macroblock_height, s->macroblock_count);

    debug_init("  %d fragments, %d x %d, u starts @ %d, v starts @ %d\n",

        s->fragment_count,

        s->fragment_width,

        s->fragment_height,

        s->u_fragment_start,

        s->v_fragment_start);



    s->all_fragments = av_malloc(s->fragment_count * sizeof(Vp3Fragment));

    s->coded_fragment_list = av_malloc(s->fragment_count * sizeof(int));

    s->pixel_addresses_inited = 0;



    /* init VLC tables */

    for (i = 0; i < 16; i++) {



        /* Dc histograms */

        init_vlc(&s->dc_vlc[i], 5, 32,

            &dc_bias[i][0][1], 4, 2,

            &dc_bias[i][0][0], 4, 2);



        /* level 1 AC histograms */

        init_vlc(&s->ac_vlc_1[i], 5, 32,

            &ac_bias_0[i][0][1], 4, 2,

            &ac_bias_0[i][0][0], 4, 2);



        /* level 2 AC histograms */

        init_vlc(&s->ac_vlc_2[i], 5, 32,

            &ac_bias_1[i][0][1], 4, 2,

            &ac_bias_1[i][0][0], 4, 2);



        /* level 3 AC histograms */

        init_vlc(&s->ac_vlc_3[i], 5, 32,

            &ac_bias_2[i][0][1], 4, 2,

            &ac_bias_2[i][0][0], 4, 2);



        /* level 4 AC histograms */

        init_vlc(&s->ac_vlc_4[i], 5, 32,

            &ac_bias_3[i][0][1], 4, 2,

            &ac_bias_3[i][0][0], 4, 2);

    }



    /* build quantization table */

    for (i = 0; i < 64; i++)

        quant_index[dequant_index[i]] = i;



    /* work out the block mapping tables */

    s->superblock_fragments = av_malloc(s->superblock_count * 16 * sizeof(int));

    s->superblock_macroblocks = av_malloc(s->superblock_count * 4 * sizeof(int));

    s->macroblock_fragments = av_malloc(s->macroblock_count * 6 * sizeof(int));

    s->macroblock_coded = av_malloc(s->macroblock_count + 1);

    init_block_mapping(s);



    /* make sure that frames are available to be freed on the first decode */

    if(avctx->get_buffer(avctx, &s->golden_frame) < 0) {

        printf("vp3: get_buffer() failed\n");

        return -1;

    }



    return 0;

}
