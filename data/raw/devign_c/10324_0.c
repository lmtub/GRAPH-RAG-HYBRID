static av_cold int vaapi_encode_h265_init_internal(AVCodecContext *avctx)

{

    static const VAConfigAttrib default_config_attributes[] = {

        { .type  = VAConfigAttribRTFormat,

          .value = VA_RT_FORMAT_YUV420 },

        { .type  = VAConfigAttribEncPackedHeaders,

          .value = (VA_ENC_PACKED_HEADER_SEQUENCE |

                    VA_ENC_PACKED_HEADER_SLICE) },

    };



    VAAPIEncodeContext      *ctx = avctx->priv_data;

    VAAPIEncodeH265Context *priv = ctx->priv_data;

    int i, err;



    switch (avctx->profile) {

    case FF_PROFILE_HEVC_MAIN:

    case FF_PROFILE_UNKNOWN:

        ctx->va_profile = VAProfileHEVCMain;

        break;

    case FF_PROFILE_HEVC_MAIN_10:

        av_log(avctx, AV_LOG_ERROR, "H.265 main 10-bit profile "

               "is not supported.\n");

        return AVERROR_PATCHWELCOME;

    default:

        av_log(avctx, AV_LOG_ERROR, "Unknown H.265 profile %d.\n",

               avctx->profile);

        return AVERROR(EINVAL);

    }

    ctx->va_entrypoint = VAEntrypointEncSlice;



    ctx->input_width    = avctx->width;

    ctx->input_height   = avctx->height;

    ctx->aligned_width  = FFALIGN(ctx->input_width,  16);

    ctx->aligned_height = FFALIGN(ctx->input_height, 16);

    priv->ctu_width     = FFALIGN(ctx->aligned_width,  32) / 32;

    priv->ctu_height    = FFALIGN(ctx->aligned_height, 32) / 32;



    av_log(avctx, AV_LOG_VERBOSE, "Input %ux%u -> Aligned %ux%u -> CTU %ux%u.\n",

           ctx->input_width, ctx->input_height, ctx->aligned_width,

           ctx->aligned_height, priv->ctu_width, priv->ctu_height);



    for (i = 0; i < FF_ARRAY_ELEMS(default_config_attributes); i++) {

        ctx->config_attributes[ctx->nb_config_attributes++] =

            default_config_attributes[i];

    }



    if (avctx->bit_rate > 0) {

        ctx->va_rc_mode = VA_RC_CBR;

        err = vaapi_encode_h265_init_constant_bitrate(avctx);

    } else {

        ctx->va_rc_mode = VA_RC_CQP;

        err = vaapi_encode_h265_init_fixed_qp(avctx);

    }

    if (err < 0)

        return err;



    ctx->config_attributes[ctx->nb_config_attributes++] = (VAConfigAttrib) {

        .type  = VAConfigAttribRateControl,

        .value = ctx->va_rc_mode,

    };



    ctx->nb_recon_frames = 20;



    return 0;

}
