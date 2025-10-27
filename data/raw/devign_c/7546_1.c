av_cold int ff_psy_init(FFPsyContext *ctx, AVCodecContext *avctx, int num_lens,

                        const uint8_t **bands, const int* num_bands,

                        int num_groups, const uint8_t *group_map)

{

    int i, j, k = 0;



    ctx->avctx = avctx;

    ctx->ch        = av_mallocz_array(sizeof(ctx->ch[0]), avctx->channels * 2);

    ctx->group     = av_mallocz_array(sizeof(ctx->group[0]), num_groups);

    ctx->bands     = av_malloc_array (sizeof(ctx->bands[0]),      num_lens);

    ctx->num_bands = av_malloc_array (sizeof(ctx->num_bands[0]),  num_lens);




    if (!ctx->ch || !ctx->group || !ctx->bands || !ctx->num_bands) {

        ff_psy_end(ctx);

        return AVERROR(ENOMEM);

    }



    memcpy(ctx->bands,     bands,     sizeof(ctx->bands[0])     *  num_lens);

    memcpy(ctx->num_bands, num_bands, sizeof(ctx->num_bands[0]) *  num_lens);



    /* assign channels to groups (with virtual channels for coupling) */

    for (i = 0; i < num_groups; i++) {

        /* NOTE: Add 1 to handle the AAC chan_config without modification.

         *       This has the side effect of allowing an array of 0s to map

         *       to one channel per group.

         */

        ctx->group[i].num_ch = group_map[i] + 1;

        for (j = 0; j < ctx->group[i].num_ch * 2; j++)

            ctx->group[i].ch[j]  = &ctx->ch[k++];

    }



    switch (ctx->avctx->codec_id) {

    case AV_CODEC_ID_AAC:

        ctx->model = &ff_aac_psy_model;

        break;

    }

    if (ctx->model->init)

        return ctx->model->init(ctx);

    return 0;

}