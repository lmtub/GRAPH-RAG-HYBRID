static av_cold int vaapi_encode_h264_init_internal(AVCodecContext *avctx)

{

    static const VAConfigAttrib default_config_attributes[] = {

        { .type  = VAConfigAttribRTFormat,

          .value = VA_RT_FORMAT_YUV420 },

        { .type  = VAConfigAttribEncPackedHeaders,

          .value = (VA_ENC_PACKED_HEADER_SEQUENCE |

                    VA_ENC_PACKED_HEADER_SLICE) },

    };



    VAAPIEncodeContext      *ctx = avctx->priv_data;

    VAAPIEncodeH264Context *priv = ctx->priv_data;

    VAAPIEncodeH264Options  *opt = ctx->codec_options;

    int i, err;



    switch (avctx->profile) {

    case FF_PROFILE_H264_CONSTRAINED_BASELINE:

        ctx->va_profile = VAProfileH264ConstrainedBaseline;

        break;

    case FF_PROFILE_H264_BASELINE:

        ctx->va_profile = VAProfileH264Baseline;

        break;

    case FF_PROFILE_H264_MAIN:

        ctx->va_profile = VAProfileH264Main;

        break;

    case FF_PROFILE_H264_EXTENDED:

        av_log(avctx, AV_LOG_ERROR, "H.264 extended profile "

               "is not supported.\n");

        return AVERROR_PATCHWELCOME;

    case FF_PROFILE_UNKNOWN:

    case FF_PROFILE_H264_HIGH:

        ctx->va_profile = VAProfileH264High;

        break;

    case FF_PROFILE_H264_HIGH_10:

    case FF_PROFILE_H264_HIGH_10_INTRA:

        av_log(avctx, AV_LOG_ERROR, "H.264 10-bit profiles "

               "are not supported.\n");

        return AVERROR_PATCHWELCOME;

    case FF_PROFILE_H264_HIGH_422:

    case FF_PROFILE_H264_HIGH_422_INTRA:

    case FF_PROFILE_H264_HIGH_444:

    case FF_PROFILE_H264_HIGH_444_PREDICTIVE:

    case FF_PROFILE_H264_HIGH_444_INTRA:

    case FF_PROFILE_H264_CAVLC_444:

        av_log(avctx, AV_LOG_ERROR, "H.264 non-4:2:0 profiles "

               "are not supported.\n");

        return AVERROR_PATCHWELCOME;

    default:

        av_log(avctx, AV_LOG_ERROR, "Unknown H.264 profile %d.\n",

               avctx->profile);

        return AVERROR(EINVAL);

    }

    if (opt->low_power) {

#if VA_CHECK_VERSION(0, 39, 1)

        ctx->va_entrypoint = VAEntrypointEncSliceLP;

#else

        av_log(avctx, AV_LOG_ERROR, "Low-power encoding is not "

               "supported with this VAAPI version.\n");

        return AVERROR(EINVAL);

#endif

    } else {

        ctx->va_entrypoint = VAEntrypointEncSlice;

    }



    ctx->input_width    = avctx->width;

    ctx->input_height   = avctx->height;

    ctx->aligned_width  = FFALIGN(ctx->input_width,  16);

    ctx->aligned_height = FFALIGN(ctx->input_height, 16);

    priv->mb_width      = ctx->aligned_width  / 16;

    priv->mb_height     = ctx->aligned_height / 16;



    for (i = 0; i < FF_ARRAY_ELEMS(default_config_attributes); i++) {

        ctx->config_attributes[ctx->nb_config_attributes++] =

            default_config_attributes[i];

    }



    if (avctx->bit_rate > 0) {

        ctx->va_rc_mode = VA_RC_CBR;

        err = vaapi_encode_h264_init_constant_bitrate(avctx);

    } else {

        ctx->va_rc_mode = VA_RC_CQP;

        err = vaapi_encode_h264_init_fixed_qp(avctx);

    }

    if (err < 0)

        return err;



    ctx->config_attributes[ctx->nb_config_attributes++] = (VAConfigAttrib) {

        .type  = VAConfigAttribRateControl,

        .value = ctx->va_rc_mode,

    };



    if (opt->quality > 0) {

#if VA_CHECK_VERSION(0, 36, 0)

        priv->quality_params.misc.type =

            VAEncMiscParameterTypeQualityLevel;

        priv->quality_params.quality.quality_level = opt->quality;



        ctx->global_params[ctx->nb_global_params] =

            &priv->quality_params.misc;

        ctx->global_params_size[ctx->nb_global_params++] =

            sizeof(priv->quality_params);

#else

        av_log(avctx, AV_LOG_WARNING, "The encode quality option is not "

               "supported with this VAAPI version.\n");

#endif

    }



    ctx->nb_recon_frames = 20;



    return 0;

}
