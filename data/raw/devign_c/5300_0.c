static av_cold int a64multi_encode_init(AVCodecContext *avctx)

{

    A64Context *c = avctx->priv_data;

    int a;

    av_lfg_init(&c->randctx, 1);



    if (avctx->global_quality < 1) {

        c->mc_lifetime = 4;

    } else {

        c->mc_lifetime = avctx->global_quality /= FF_QP2LAMBDA;

    }



    av_log(avctx, AV_LOG_INFO, "charset lifetime set to %d frame(s)\n", c->mc_lifetime);



    c->mc_frame_counter = 0;

    c->mc_use_5col      = avctx->codec->id == AV_CODEC_ID_A64_MULTI5;

    c->mc_pal_size      = 4 + c->mc_use_5col;



    /* precalc luma values for later use */

    for (a = 0; a < c->mc_pal_size; a++) {

        c->mc_luma_vals[a]=a64_palette[mc_colors[a]][0] * 0.30 +

                           a64_palette[mc_colors[a]][1] * 0.59 +

                           a64_palette[mc_colors[a]][2] * 0.11;

    }



    if (!(c->mc_meta_charset = av_malloc(32000 * c->mc_lifetime * sizeof(int))) ||

       !(c->mc_best_cb       = av_malloc(CHARSET_CHARS * 32 * sizeof(int)))     ||

       !(c->mc_charmap       = av_mallocz(1000 * c->mc_lifetime * sizeof(int))) ||

       !(c->mc_colram        = av_mallocz(CHARSET_CHARS * sizeof(uint8_t)))     ||

       !(c->mc_charset       = av_malloc(0x800 * (INTERLACED+1) * sizeof(uint8_t)))) {

        av_log(avctx, AV_LOG_ERROR, "Failed to allocate buffer memory.\n");

        return AVERROR(ENOMEM);

    }



    /* set up extradata */

    if (!(avctx->extradata = av_mallocz(8 * 4 + FF_INPUT_BUFFER_PADDING_SIZE))) {

        av_log(avctx, AV_LOG_ERROR, "Failed to allocate memory for extradata.\n");

        return AVERROR(ENOMEM);

    }

    avctx->extradata_size = 8 * 4;

    AV_WB32(avctx->extradata, c->mc_lifetime);

    AV_WB32(avctx->extradata + 16, INTERLACED);



    avctx->coded_frame = av_frame_alloc();

    if (!avctx->coded_frame) {

        a64multi_close_encoder(avctx);

        return AVERROR(ENOMEM);

    }



    avctx->coded_frame->pict_type = AV_PICTURE_TYPE_I;

    avctx->coded_frame->key_frame = 1;

    if (!avctx->codec_tag)

         avctx->codec_tag = AV_RL32("a64m");



    c->next_pts = AV_NOPTS_VALUE;



    return 0;

}
