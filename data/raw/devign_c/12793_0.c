static int vaapi_decode_make_config(AVCodecContext *avctx)

{

    VAAPIDecodeContext *ctx = avctx->internal->hwaccel_priv_data;



    AVVAAPIHWConfig       *hwconfig    = NULL;

    AVHWFramesConstraints *constraints = NULL;

    VAStatus vas;

    int err, i, j;

    const AVCodecDescriptor *codec_desc;

    VAProfile profile, *profile_list = NULL;

    int profile_count, exact_match, alt_profile;

    const AVPixFmtDescriptor *sw_desc, *desc;



    codec_desc = avcodec_descriptor_get(avctx->codec_id);

    if (!codec_desc) {

        err = AVERROR(EINVAL);

        goto fail;

    }



    profile_count = vaMaxNumProfiles(ctx->hwctx->display);

    profile_list  = av_malloc_array(profile_count,

                                    sizeof(VAProfile));

    if (!profile_list) {

        err = AVERROR(ENOMEM);

        goto fail;

    }



    vas = vaQueryConfigProfiles(ctx->hwctx->display,

                                profile_list, &profile_count);

    if (vas != VA_STATUS_SUCCESS) {

        av_log(avctx, AV_LOG_ERROR, "Failed to query profiles: "

               "%d (%s).\n", vas, vaErrorStr(vas));

        err = AVERROR(ENOSYS);

        goto fail;

    }



    profile = VAProfileNone;

    exact_match = 0;



    for (i = 0; i < FF_ARRAY_ELEMS(vaapi_profile_map); i++) {

        int profile_match = 0;

        if (avctx->codec_id != vaapi_profile_map[i].codec_id)

            continue;

        if (avctx->profile == vaapi_profile_map[i].codec_profile ||

            vaapi_profile_map[i].codec_profile == FF_PROFILE_UNKNOWN)

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

        av_log(avctx, AV_LOG_ERROR, "No support for codec %s "

               "profile %d.\n", codec_desc->name, avctx->profile);

        err = AVERROR(ENOSYS);

        goto fail;

    }

    if (!exact_match) {

        if (avctx->hwaccel_flags &

            AV_HWACCEL_FLAG_ALLOW_PROFILE_MISMATCH) {

            av_log(avctx, AV_LOG_VERBOSE, "Codec %s profile %d not "

                   "supported for hardware decode.\n",

                   codec_desc->name, avctx->profile);

            av_log(avctx, AV_LOG_WARNING, "Using possibly-"

                   "incompatible profile %d instead.\n",

                   alt_profile);

        } else {

            av_log(avctx, AV_LOG_VERBOSE, "Codec %s profile %d not "

                   "supported for hardware decode.\n",

                   codec_desc->name, avctx->profile);

            err = AVERROR(EINVAL);

            goto fail;

        }

    }



    ctx->va_profile    = profile;

    ctx->va_entrypoint = VAEntrypointVLD;



    vas = vaCreateConfig(ctx->hwctx->display, ctx->va_profile,

                         ctx->va_entrypoint, NULL, 0,

                         &ctx->va_config);

    if (vas != VA_STATUS_SUCCESS) {

        av_log(avctx, AV_LOG_ERROR, "Failed to create decode "

               "configuration: %d (%s).\n", vas, vaErrorStr(vas));

        err = AVERROR(EIO);

        goto fail;

    }



    hwconfig = av_hwdevice_hwconfig_alloc(avctx->hw_device_ctx ?

                                          avctx->hw_device_ctx :

                                          ctx->frames->device_ref);

    if (!hwconfig) {

        err = AVERROR(ENOMEM);

        goto fail;

    }

    hwconfig->config_id = ctx->va_config;



    constraints =

        av_hwdevice_get_hwframe_constraints(avctx->hw_device_ctx ?

                                            avctx->hw_device_ctx :

                                            ctx->frames->device_ref,

                                            hwconfig);

    if (!constraints) {

        err = AVERROR(ENOMEM);

        goto fail;

    }



    if (avctx->coded_width  < constraints->min_width  ||

        avctx->coded_height < constraints->min_height ||

        avctx->coded_width  > constraints->max_width  ||

        avctx->coded_height > constraints->max_height) {

        av_log(avctx, AV_LOG_ERROR, "Hardware does not support image "

               "size %dx%d (constraints: width %d-%d height %d-%d).\n",

               avctx->coded_width, avctx->coded_height,

               constraints->min_width,  constraints->max_width,

               constraints->min_height, constraints->max_height);

        err = AVERROR(EINVAL);

        goto fail;

    }

    if (!constraints->valid_sw_formats ||

        constraints->valid_sw_formats[0] == AV_PIX_FMT_NONE) {

        av_log(avctx, AV_LOG_ERROR, "Hardware does not offer any "

               "usable surface formats.\n");

        err = AVERROR(EINVAL);

        goto fail;

    }



    // Find the first format in the list which matches the expected

    // bit depth and subsampling.  If none are found (this can happen

    // when 10-bit streams are decoded to 8-bit surfaces, for example)

    // then just take the first format on the list.

    ctx->surface_format = constraints->valid_sw_formats[0];

    sw_desc = av_pix_fmt_desc_get(avctx->sw_pix_fmt);

    for (i = 0; constraints->valid_sw_formats[i] != AV_PIX_FMT_NONE; i++) {

        desc = av_pix_fmt_desc_get(constraints->valid_sw_formats[i]);

        if (desc->nb_components != sw_desc->nb_components ||

            desc->log2_chroma_w != sw_desc->log2_chroma_w ||

            desc->log2_chroma_h != sw_desc->log2_chroma_h)

            continue;

        for (j = 0; j < desc->nb_components; j++) {

            if (desc->comp[j].depth != sw_desc->comp[j].depth)

                break;

        }

        if (j < desc->nb_components)

            continue;

        ctx->surface_format = constraints->valid_sw_formats[i];

        break;

    }



    // Start with at least four surfaces.

    ctx->surface_count = 4;

    // Add per-codec number of surfaces used for storing reference frames.

    switch (avctx->codec_id) {

    case AV_CODEC_ID_H264:

    case AV_CODEC_ID_HEVC:

        ctx->surface_count += 16;

        break;

    case AV_CODEC_ID_VP9:

        ctx->surface_count += 8;

        break;

    case AV_CODEC_ID_VP8:

        ctx->surface_count += 3;

        break;

    default:

        ctx->surface_count += 2;

    }

    // Add an additional surface per thread is frame threading is enabled.

    if (avctx->active_thread_type & FF_THREAD_FRAME)

        ctx->surface_count += avctx->thread_count;



    av_hwframe_constraints_free(&constraints);

    av_freep(&hwconfig);



    return 0;



fail:

    av_hwframe_constraints_free(&constraints);

    av_freep(&hwconfig);

    if (ctx->va_config != VA_INVALID_ID) {

        vaDestroyConfig(ctx->hwctx->display, ctx->va_config);

        ctx->va_config = VA_INVALID_ID;

    }

    av_freep(&profile_list);

    return err;

}
