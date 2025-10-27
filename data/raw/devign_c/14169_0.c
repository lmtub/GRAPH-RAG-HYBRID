static av_cold int vaapi_encode_h265_init_constant_bitrate(AVCodecContext *avctx)

{

    VAAPIEncodeContext      *ctx = avctx->priv_data;

    VAAPIEncodeH265Context *priv = ctx->priv_data;

    int hrd_buffer_size;

    int hrd_initial_buffer_fullness;



    if (avctx->bit_rate > INT32_MAX) {

        av_log(avctx, AV_LOG_ERROR, "Target bitrate of 2^31 bps or "

               "higher is not supported.\n");

        return AVERROR(EINVAL);

    }



    if (avctx->rc_buffer_size)

        hrd_buffer_size = avctx->rc_buffer_size;

    else

        hrd_buffer_size = avctx->bit_rate;

    if (avctx->rc_initial_buffer_occupancy)

        hrd_initial_buffer_fullness = avctx->rc_initial_buffer_occupancy;

    else

        hrd_initial_buffer_fullness = hrd_buffer_size * 3 / 4;



    priv->rc_params.misc.type = VAEncMiscParameterTypeRateControl;

    priv->rc_params.rc = (VAEncMiscParameterRateControl) {

        .bits_per_second   = avctx->bit_rate,

        .target_percentage = 66,

        .window_size       = 1000,

        .initial_qp        = (avctx->qmax >= 0 ? avctx->qmax : 40),

        .min_qp            = (avctx->qmin >= 0 ? avctx->qmin : 20),

        .basic_unit_size   = 0,

    };

    ctx->global_params[ctx->nb_global_params] =

        &priv->rc_params.misc;

    ctx->global_params_size[ctx->nb_global_params++] =

        sizeof(priv->rc_params);



    priv->hrd_params.misc.type = VAEncMiscParameterTypeHRD;

    priv->hrd_params.hrd = (VAEncMiscParameterHRD) {

        .initial_buffer_fullness = hrd_initial_buffer_fullness,

        .buffer_size             = hrd_buffer_size,

    };

    ctx->global_params[ctx->nb_global_params] =

        &priv->hrd_params.misc;

    ctx->global_params_size[ctx->nb_global_params++] =

        sizeof(priv->hrd_params);



    // These still need to be  set for pic_init_qp/slice_qp_delta.

    priv->fixed_qp_idr = 30;

    priv->fixed_qp_p   = 30;

    priv->fixed_qp_b   = 30;



    av_log(avctx, AV_LOG_DEBUG, "Using constant-bitrate = %"PRId64" bps.\n",

           avctx->bit_rate);

    return 0;

}
