static av_cold int nvenc_recalc_surfaces(AVCodecContext *avctx)

{

    NvencContext *ctx = avctx->priv_data;

    int nb_surfaces = 0;



    if (ctx->rc_lookahead > 0) {

        nb_surfaces = ctx->rc_lookahead + ((ctx->encode_config.frameIntervalP > 0) ? ctx->encode_config.frameIntervalP : 0) + 1 + 4;

        if (ctx->nb_surfaces < nb_surfaces) {

            av_log(avctx, AV_LOG_WARNING,

                   "Defined rc_lookahead requires more surfaces, "

                   "increasing used surfaces %d -> %d\n", ctx->nb_surfaces, nb_surfaces);

            ctx->nb_surfaces = nb_surfaces;

        }

    }



    ctx->nb_surfaces = FFMAX(1, FFMIN(MAX_REGISTERED_FRAMES, ctx->nb_surfaces));

    ctx->async_depth = FFMIN(ctx->async_depth, ctx->nb_surfaces - 1);



    return 0;

}
