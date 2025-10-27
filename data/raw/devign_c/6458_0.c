static av_cold int vaapi_encode_check_config(AVCodecContext *avctx)

{

    VAAPIEncodeContext *ctx = avctx->priv_data;

    VAStatus vas;

    int i, n, err;

    VAProfile    *profiles    = NULL;

    VAEntrypoint *entrypoints = NULL;

    VAConfigAttrib attr[] = {

        { VAConfigAttribRateControl     },

        { VAConfigAttribEncMaxRefFrames },

    };



    n = vaMaxNumProfiles(ctx->hwctx->display);

    profiles = av_malloc_array(n, sizeof(VAProfile));

    if (!profiles) {

        err = AVERROR(ENOMEM);

        goto fail;

    }

    vas = vaQueryConfigProfiles(ctx->hwctx->display, profiles, &n);

    if (vas != VA_STATUS_SUCCESS) {

        av_log(ctx, AV_LOG_ERROR, "Failed to query profiles: %d (%s).\n",

               vas, vaErrorStr(vas));

        err = AVERROR(ENOSYS);

        goto fail;

    }

    for (i = 0; i < n; i++) {

        if (profiles[i] == ctx->va_profile)

            break;

    }

    if (i >= n) {

        av_log(ctx, AV_LOG_ERROR, "Encoding profile not found (%d).\n",

               ctx->va_profile);

        err = AVERROR(ENOSYS);

        goto fail;

    }



    n = vaMaxNumEntrypoints(ctx->hwctx->display);

    entrypoints = av_malloc_array(n, sizeof(VAEntrypoint));

    if (!entrypoints) {

        err = AVERROR(ENOMEM);

        goto fail;

    }

    vas = vaQueryConfigEntrypoints(ctx->hwctx->display, ctx->va_profile,

                                   entrypoints, &n);

    if (vas != VA_STATUS_SUCCESS) {

        av_log(ctx, AV_LOG_ERROR, "Failed to query entrypoints for "

               "profile %u: %d (%s).\n", ctx->va_profile,

               vas, vaErrorStr(vas));

        err = AVERROR(ENOSYS);

        goto fail;

    }

    for (i = 0; i < n; i++) {

        if (entrypoints[i] == ctx->va_entrypoint)

            break;

    }

    if (i >= n) {

        av_log(ctx, AV_LOG_ERROR, "Encoding entrypoint not found "

               "(%d / %d).\n", ctx->va_profile, ctx->va_entrypoint);

        err = AVERROR(ENOSYS);

        goto fail;

    }



    vas = vaGetConfigAttributes(ctx->hwctx->display,

                                ctx->va_profile, ctx->va_entrypoint,

                                attr, FF_ARRAY_ELEMS(attr));

    if (vas != VA_STATUS_SUCCESS) {

        av_log(avctx, AV_LOG_ERROR, "Failed to fetch config "

               "attributes: %d (%s).\n", vas, vaErrorStr(vas));

        return AVERROR(EINVAL);

    }



    for (i = 0; i < FF_ARRAY_ELEMS(attr); i++) {

        if (attr[i].value == VA_ATTRIB_NOT_SUPPORTED) {

            // Unfortunately we have to treat this as "don't know" and hope

            // for the best, because the Intel MJPEG encoder returns this

            // for all the interesting attributes.

            continue;

        }

        switch (attr[i].type) {

        case VAConfigAttribRateControl:

            if (!(ctx->va_rc_mode & attr[i].value)) {

                av_log(avctx, AV_LOG_ERROR, "Rate control mode is not "

                       "supported: %x\n", attr[i].value);

                err = AVERROR(EINVAL);

                goto fail;

            }

            break;

        case VAConfigAttribEncMaxRefFrames:

        {

            unsigned int ref_l0 = attr[i].value & 0xffff;

            unsigned int ref_l1 = (attr[i].value >> 16) & 0xffff;



            if (avctx->gop_size > 1 && ref_l0 < 1) {

                av_log(avctx, AV_LOG_ERROR, "P frames are not "

                       "supported (%x).\n", attr[i].value);

                err = AVERROR(EINVAL);

                goto fail;

            }

            if (avctx->max_b_frames > 0 && ref_l1 < 1) {

                av_log(avctx, AV_LOG_ERROR, "B frames are not "

                       "supported (%x).\n", attr[i].value);

                err = AVERROR(EINVAL);

                goto fail;

            }

        }

        break;

        }

    }



    err = 0;

fail:

    av_freep(&profiles);

    av_freep(&entrypoints);

    return err;

}
