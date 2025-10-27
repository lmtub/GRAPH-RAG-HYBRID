static int dxva2_alloc(AVCodecContext *s)

{

    InputStream  *ist = s->opaque;

    int loglevel = (ist->hwaccel_id == HWACCEL_AUTO) ? AV_LOG_VERBOSE : AV_LOG_ERROR;

    DXVA2Context *ctx;

    HANDLE device_handle;

    HRESULT hr;



    AVHWDeviceContext    *device_ctx;

    AVDXVA2DeviceContext *device_hwctx;

    int ret;



    ctx = av_mallocz(sizeof(*ctx));

    if (!ctx)

        return AVERROR(ENOMEM);



    ist->hwaccel_ctx           = ctx;

    ist->hwaccel_uninit        = dxva2_uninit;

    ist->hwaccel_get_buffer    = dxva2_get_buffer;

    ist->hwaccel_retrieve_data = dxva2_retrieve_data;



    ret = av_hwdevice_ctx_create(&ctx->hw_device_ctx, AV_HWDEVICE_TYPE_DXVA2,

                                 ist->hwaccel_device, NULL, 0);

    if (ret < 0)

        goto fail;

    device_ctx   = (AVHWDeviceContext*)ctx->hw_device_ctx->data;

    device_hwctx = device_ctx->hwctx;



    hr = IDirect3DDeviceManager9_OpenDeviceHandle(device_hwctx->devmgr,

                                                  &device_handle);

    if (FAILED(hr)) {

        av_log(NULL, loglevel, "Failed to open a device handle\n");

        goto fail;

    }



    hr = IDirect3DDeviceManager9_GetVideoService(device_hwctx->devmgr, device_handle,

                                                 &IID_IDirectXVideoDecoderService,

                                                 (void **)&ctx->decoder_service);

    IDirect3DDeviceManager9_CloseDeviceHandle(device_hwctx->devmgr, device_handle);

    if (FAILED(hr)) {

        av_log(NULL, loglevel, "Failed to create IDirectXVideoDecoderService\n");

        goto fail;

    }



    ctx->tmp_frame = av_frame_alloc();

    if (!ctx->tmp_frame)

        goto fail;



    s->hwaccel_context = av_mallocz(sizeof(struct dxva_context));

    if (!s->hwaccel_context)

        goto fail;



    return 0;

fail:

    dxva2_uninit(s);

    return AVERROR(EINVAL);

}
