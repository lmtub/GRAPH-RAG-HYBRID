int ff_vaapi_decode_init(AVCodecContext *avctx)

{

    VAAPIDecodeContext *ctx = avctx->internal->hwaccel_priv_data;

    VAStatus vas;

    int err;



    ctx->va_config  = VA_INVALID_ID;

    ctx->va_context = VA_INVALID_ID;



#if FF_API_VAAPI_CONTEXT

    if (avctx->hwaccel_context) {

        av_log(avctx, AV_LOG_WARNING, "Using deprecated struct "

               "vaapi_context in decode.\n");



        ctx->have_old_context = 1;

        ctx->old_context = avctx->hwaccel_context;



        // Really we only want the VAAPI device context, but this

        // allocates a whole generic device context because we don't

        // have any other way to determine how big it should be.

        ctx->device_ref =

            av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_VAAPI);

        if (!ctx->device_ref) {

            err = AVERROR(ENOMEM);

            goto fail;

        }

        ctx->device = (AVHWDeviceContext*)ctx->device_ref->data;

        ctx->hwctx  = ctx->device->hwctx;



        ctx->hwctx->display = ctx->old_context->display;



        // The old VAAPI decode setup assumed this quirk was always

        // present, so set it here to avoid the behaviour changing.

        ctx->hwctx->driver_quirks =

            AV_VAAPI_DRIVER_QUIRK_RENDER_PARAM_BUFFERS;



    } else

#endif

    if (avctx->hw_frames_ctx) {

        // This structure has a shorter lifetime than the enclosing

        // AVCodecContext, so we inherit the references from there

        // and do not need to make separate ones.



        ctx->frames = (AVHWFramesContext*)avctx->hw_frames_ctx->data;

        ctx->hwfc   = ctx->frames->hwctx;

        ctx->device = ctx->frames->device_ctx;

        ctx->hwctx  = ctx->device->hwctx;



    } else if (avctx->hw_device_ctx) {

        ctx->device = (AVHWDeviceContext*)avctx->hw_device_ctx->data;

        ctx->hwctx  = ctx->device->hwctx;



        if (ctx->device->type != AV_HWDEVICE_TYPE_VAAPI) {

            av_log(avctx, AV_LOG_ERROR, "Device supplied for VAAPI "

                   "decoding must be a VAAPI device (not %d).\n",

                   ctx->device->type);

            err = AVERROR(EINVAL);

            goto fail;

        }



    } else {

        av_log(avctx, AV_LOG_ERROR, "A hardware device or frames context "

               "is required for VAAPI decoding.\n");

        err = AVERROR(EINVAL);

        goto fail;

    }



#if FF_API_VAAPI_CONTEXT

    if (ctx->have_old_context) {

        ctx->va_config  = ctx->old_context->config_id;

        ctx->va_context = ctx->old_context->context_id;



        av_log(avctx, AV_LOG_DEBUG, "Using user-supplied decoder "

               "context: %#x/%#x.\n", ctx->va_config, ctx->va_context);

    } else {

#endif



    err = vaapi_decode_make_config(avctx);

    if (err)

        goto fail;



    if (!avctx->hw_frames_ctx) {

        avctx->hw_frames_ctx = av_hwframe_ctx_alloc(avctx->hw_device_ctx);

        if (!avctx->hw_frames_ctx) {

            err = AVERROR(ENOMEM);

            goto fail;

        }

        ctx->frames = (AVHWFramesContext*)avctx->hw_frames_ctx->data;



        ctx->frames->format = AV_PIX_FMT_VAAPI;

        ctx->frames->width  = avctx->coded_width;

        ctx->frames->height = avctx->coded_height;



        ctx->frames->sw_format         = ctx->surface_format;

        ctx->frames->initial_pool_size = ctx->surface_count;



        err = av_hwframe_ctx_init(avctx->hw_frames_ctx);

        if (err < 0) {

            av_log(avctx, AV_LOG_ERROR, "Failed to initialise internal "

                   "frames context: %d.\n", err);

            goto fail;

        }



        ctx->hwfc = ctx->frames->hwctx;

    }



    vas = vaCreateContext(ctx->hwctx->display, ctx->va_config,

                          avctx->coded_width, avctx->coded_height,

                          VA_PROGRESSIVE,

                          ctx->hwfc->surface_ids,

                          ctx->hwfc->nb_surfaces,

                          &ctx->va_context);

    if (vas != VA_STATUS_SUCCESS) {

        av_log(avctx, AV_LOG_ERROR, "Failed to create decode "

               "context: %d (%s).\n", vas, vaErrorStr(vas));

        err = AVERROR(EIO);

        goto fail;

    }



    av_log(avctx, AV_LOG_DEBUG, "Decode context initialised: "

           "%#x/%#x.\n", ctx->va_config, ctx->va_context);

#if FF_API_VAAPI_CONTEXT

    }

#endif



    return 0;



fail:

    ff_vaapi_decode_uninit(avctx);

    return err;

}
