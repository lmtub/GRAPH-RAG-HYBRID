static int vaapi_build_decoder_config(VAAPIDecoderContext *ctx,

                                      AVCodecContext *avctx,

                                      int fallback_allowed)

{

    AVVAAPIDeviceContext *hwctx = ctx->device->hwctx;

    AVVAAPIHWConfig *hwconfig = NULL;

    AVHWFramesConstraints *constraints = NULL;

    VAStatus vas;

    int err, i, j;

    int loglevel = fallback_allowed ? AV_LOG_VERBOSE : AV_LOG_ERROR;

    const AVCodecDescriptor *codec_desc;

    const AVPixFmtDescriptor *pix_desc;

    enum AVPixelFormat pix_fmt;

    VAProfile profile, *profile_list = NULL;

    int profile_count, exact_match, alt_profile;



    codec_desc = avcodec_descriptor_get(avctx->codec_id);

    if (!codec_desc) {

        err = AVERROR(EINVAL);

        goto fail;

    }



    profile_count = vaMaxNumProfiles(hwctx->display);

    profile_list = av_malloc(profile_count * sizeof(VAProfile));

    if (!profile_list) {

        err = AVERROR(ENOMEM);

        goto fail;

    }



    vas = vaQueryConfigProfiles(hwctx->display,

                                profile_list, &profile_count);

    if (vas != VA_STATUS_SUCCESS) {

        av_log(ctx, loglevel, "Failed to query profiles: %d (%s).\n",

               vas, vaErrorStr(vas));

        err = AVERROR(EIO);

        goto fail;

    }



    profile = VAProfileNone;

    exact_match = 0;



    for (i = 0; i < FF_ARRAY_ELEMS(vaapi_profile_map); i++) {

        int profile_match = 0;

        if (avctx->codec_id != vaapi_profile_map[i].codec_id)

            continue;

        if (avctx->profile == vaapi_profile_map[i].codec_profile)

            profile_match = 1;

        profile = vaapi_profile_map[i].va_profile;

        for (j = 0; j < profile_count; j++) {

            if (profile == profile_list[j]) {

                exact_match = profile_match;

                break;

            }

        }

        if (j < profile_count) {

            if (exact_match)

                break;

            alt_profile = vaapi_profile_map[i].codec_profile;

        }

    }

    av_freep(&profile_list);



    if (profile == VAProfileNone) {

        av_log(ctx, loglevel, "No VAAPI support for codec %s.\n",

               codec_desc->name);

        err = AVERROR(ENOSYS);

        goto fail;

    }

    if (!exact_match) {

        if (fallback_allowed || !hwaccel_lax_profile_check) {

            av_log(ctx, loglevel, "No VAAPI support for codec %s "

                   "profile %d.\n", codec_desc->name, avctx->profile);

            if (!fallback_allowed) {

                av_log(ctx, AV_LOG_WARNING, "If you want attempt decoding "

                       "anyway with a possibly-incompatible profile, add "

                       "the option -hwaccel_lax_profile_check.\n");

            }

            err = AVERROR(EINVAL);

            goto fail;

        } else {

            av_log(ctx, AV_LOG_WARNING, "No VAAPI support for codec %s "

                   "profile %d: trying instead with profile %d.\n",

                   codec_desc->name, avctx->profile, alt_profile);

            av_log(ctx, AV_LOG_WARNING, "This may fail or give "

                   "incorrect results, depending on your hardware.\n");

        }

    }



    ctx->va_profile = profile;

    ctx->va_entrypoint = VAEntrypointVLD;



    vas = vaCreateConfig(hwctx->display, ctx->va_profile,

                         ctx->va_entrypoint, 0, 0, &ctx->va_config);

    if (vas != VA_STATUS_SUCCESS) {

        av_log(ctx, AV_LOG_ERROR, "Failed to create decode pipeline "

               "configuration: %d (%s).\n", vas, vaErrorStr(vas));

        err = AVERROR(EIO);

        goto fail;

    }



    hwconfig = av_hwdevice_hwconfig_alloc(ctx->device_ref);

    if (!hwconfig) {

        err = AVERROR(ENOMEM);

        goto fail;

    }

    hwconfig->config_id = ctx->va_config;



    constraints = av_hwdevice_get_hwframe_constraints(ctx->device_ref,

                                                      hwconfig);

    if (!constraints)

        goto fail;



    // Decide on the decoder target format.

    // If the user specified something with -hwaccel_output_format then

    // try to use that to minimise conversions later.

    ctx->decode_format = AV_PIX_FMT_NONE;

    if (ctx->output_format != AV_PIX_FMT_NONE &&

        ctx->output_format != AV_PIX_FMT_VAAPI) {

        for (i = 0; constraints->valid_sw_formats[i] != AV_PIX_FMT_NONE; i++) {

            if (constraints->valid_sw_formats[i] == ctx->decode_format) {

                ctx->decode_format = ctx->output_format;

                av_log(ctx, AV_LOG_DEBUG, "Using decode format %s (output "

                       "format).\n", av_get_pix_fmt_name(ctx->decode_format));

                break;

            }

        }

    }

    // Otherwise, we would like to try to choose something which matches the

    // decoder output, but there isn't enough information available here to

    // do so.  Assume for now that we are always dealing with YUV 4:2:0, so

    // pick a format which does that.

    if (ctx->decode_format == AV_PIX_FMT_NONE) {

        for (i = 0; constraints->valid_sw_formats[i] != AV_PIX_FMT_NONE; i++) {

            pix_fmt  = constraints->valid_sw_formats[i];

            pix_desc = av_pix_fmt_desc_get(pix_fmt);

            if (pix_desc->nb_components == 3 &&

                pix_desc->log2_chroma_w == 1 &&

                pix_desc->log2_chroma_h == 1) {

                ctx->decode_format = pix_fmt;

                av_log(ctx, AV_LOG_DEBUG, "Using decode format %s (format "

                       "matched).\n", av_get_pix_fmt_name(ctx->decode_format));

                break;

            }

        }

    }

    // Otherwise pick the first in the list and hope for the best.

    if (ctx->decode_format == AV_PIX_FMT_NONE) {

        ctx->decode_format = constraints->valid_sw_formats[0];

        av_log(ctx, AV_LOG_DEBUG, "Using decode format %s (first in list).\n",

               av_get_pix_fmt_name(ctx->decode_format));

        if (i > 1) {

            // There was a choice, and we picked randomly.  Warn the user

            // that they might want to choose intelligently instead.

            av_log(ctx, AV_LOG_WARNING, "Using randomly chosen decode "

                   "format %s.\n", av_get_pix_fmt_name(ctx->decode_format));

        }

    }



    // Ensure the picture size is supported by the hardware.

    ctx->decode_width  = avctx->coded_width;

    ctx->decode_height = avctx->coded_height;

    if (ctx->decode_width  < constraints->min_width  ||

        ctx->decode_height < constraints->min_height ||

        ctx->decode_width  > constraints->max_width  ||

        ctx->decode_height >constraints->max_height) {

        av_log(ctx, AV_LOG_ERROR, "VAAPI hardware does not support image "

               "size %dx%d (constraints: width %d-%d height %d-%d).\n",

               ctx->decode_width, ctx->decode_height,

               constraints->min_width,  constraints->max_width,

               constraints->min_height, constraints->max_height);

        err = AVERROR(EINVAL);

        goto fail;

    }



    av_hwframe_constraints_free(&constraints);

    av_freep(&hwconfig);



    // Decide how many reference frames we need.  This might be doable more

    // nicely based on the codec and input stream?

    ctx->decode_surfaces = DEFAULT_SURFACES;

    // For frame-threaded decoding, one additional surfaces is needed for

    // each thread.

    if (avctx->active_thread_type & FF_THREAD_FRAME)

        ctx->decode_surfaces += avctx->thread_count;



    return 0;



fail:

    av_hwframe_constraints_free(&constraints);

    av_freep(&hwconfig);

    vaDestroyConfig(hwctx->display, ctx->va_config);

    av_freep(&profile_list);

    return err;

}
