static av_cold int nvenc_check_cuda(AVCodecContext *avctx)

{

    int device_count = 0;

    CUdevice cu_device = 0;

    char gpu_name[128];

    int smminor = 0, smmajor = 0;

    int i, smver, target_smver;



    NvencContext *ctx = avctx->priv_data;

    NvencDynLoadFunctions *dl_fn = &ctx->nvenc_dload_funcs;



    switch (avctx->codec->id) {

    case AV_CODEC_ID_H264:

        target_smver = ctx->data_pix_fmt == AV_PIX_FMT_YUV444P ? 0x52 : 0x30;

        break;

    case AV_CODEC_ID_H265:

        target_smver = 0x52;

        break;

    default:

        av_log(avctx, AV_LOG_FATAL, "Unknown codec name\n");

        goto error;

    }



    if (ctx->preset >= PRESET_LOSSLESS_DEFAULT)

        target_smver = 0x52;



    if (!nvenc_dyload_cuda(avctx))

        return 0;



    if (dl_fn->nvenc_device_count > 0)

        return 1;



    check_cuda_errors(dl_fn->cu_init(0));



    check_cuda_errors(dl_fn->cu_device_get_count(&device_count));



    if (!device_count) {

        av_log(avctx, AV_LOG_FATAL, "No CUDA capable devices found\n");

        goto error;

    }



    av_log(avctx, AV_LOG_VERBOSE, "%d CUDA capable devices found\n", device_count);



    dl_fn->nvenc_device_count = 0;



    for (i = 0; i < device_count; ++i) {

        check_cuda_errors(dl_fn->cu_device_get(&cu_device, i));

        check_cuda_errors(dl_fn->cu_device_get_name(gpu_name, sizeof(gpu_name), cu_device));

        check_cuda_errors(dl_fn->cu_device_compute_capability(&smmajor, &smminor, cu_device));



        smver = (smmajor << 4) | smminor;



        av_log(avctx, AV_LOG_VERBOSE, "[ GPU #%d - < %s > has Compute SM %d.%d, NVENC %s ]\n", i, gpu_name, smmajor, smminor, (smver >= target_smver) ? "Available" : "Not Available");



        if (smver >= target_smver)

            dl_fn->nvenc_devices[dl_fn->nvenc_device_count++] = cu_device;

    }



    if (!dl_fn->nvenc_device_count) {

        av_log(avctx, AV_LOG_FATAL, "No NVENC capable devices found\n");

        goto error;

    }



    return 1;



error:



    dl_fn->nvenc_device_count = 0;



    return 0;

}
