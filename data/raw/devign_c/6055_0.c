static void nvenc_setup_rate_control(AVCodecContext *avctx)

{

    NVENCContext *ctx    = avctx->priv_data;

    NV_ENC_RC_PARAMS *rc = &ctx->config.rcParams;



    if (avctx->bit_rate > 0)

        rc->averageBitRate = avctx->bit_rate;



    if (avctx->rc_max_rate > 0)

        rc->maxBitRate = avctx->rc_max_rate;



    if (ctx->rc > 0) {

        nvenc_override_rate_control(avctx, rc);

    } else if (ctx->flags & NVENC_LOSSLESS) {

        set_lossless(avctx, rc);

    } else if (avctx->global_quality > 0) {

        set_constqp(avctx, rc);

    } else if (avctx->qmin >= 0 && avctx->qmax >= 0) {

        rc->rateControlMode = NV_ENC_PARAMS_RC_VBR;

        set_vbr(avctx, rc);

    }



    if (avctx->rc_buffer_size > 0)

        rc->vbvBufferSize = avctx->rc_buffer_size;



    if (rc->averageBitRate > 0)

        avctx->bit_rate = rc->averageBitRate;



#if NVENCAPI_MAJOR_VERSION >= 7

    if (ctx->aq) {

        ctx->config.rcParams.enableAQ   = 1;

        ctx->config.rcParams.aqStrength = ctx->aq_strength;

        av_log(avctx, AV_LOG_VERBOSE, "AQ enabled.\n");

    }



    if (ctx->temporal_aq) {

        ctx->config.rcParams.enableTemporalAQ = 1;

        av_log(avctx, AV_LOG_VERBOSE, "Temporal AQ enabled.\n");

    }



    if (ctx->rc_lookahead) {

        int lkd_bound = FFMIN(ctx->nb_surfaces, ctx->async_depth) -

                        ctx->config.frameIntervalP - 4;



        if (lkd_bound < 0) {

            av_log(avctx, AV_LOG_WARNING,

                   "Lookahead not enabled. Increase buffer delay (-delay).\n");

        } else {

            ctx->config.rcParams.enableLookahead = 1;

            ctx->config.rcParams.lookaheadDepth  = av_clip(ctx->rc_lookahead, 0, lkd_bound);

            ctx->config.rcParams.disableIadapt   = ctx->no_scenecut;

            ctx->config.rcParams.disableBadapt   = !ctx->b_adapt;

            av_log(avctx, AV_LOG_VERBOSE,

                   "Lookahead enabled: depth %d, scenecut %s, B-adapt %s.\n",

                   ctx->config.rcParams.lookaheadDepth,

                   ctx->config.rcParams.disableIadapt ? "disabled" : "enabled",

                   ctx->config.rcParams.disableBadapt ? "disabled" : "enabled");

        }

    }



    if (ctx->strict_gop) {

        ctx->config.rcParams.strictGOPTarget = 1;

        av_log(avctx, AV_LOG_VERBOSE, "Strict GOP target enabled.\n");

    }



    if (ctx->nonref_p)

        ctx->config.rcParams.enableNonRefP = 1;



    if (ctx->zerolatency)

        ctx->config.rcParams.zeroReorderDelay = 1;



    if (ctx->quality)

        ctx->config.rcParams.targetQuality = ctx->quality;

#endif /* NVENCAPI_MAJOR_VERSION >= 7 */

}
