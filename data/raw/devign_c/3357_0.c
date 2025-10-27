static av_cold int nvenc_alloc_surface(AVCodecContext *avctx, int idx)

{

    NvencContext *ctx = avctx->priv_data;

    NvencDynLoadFunctions *dl_fn = &ctx->nvenc_dload_funcs;

    NV_ENCODE_API_FUNCTION_LIST *p_nvenc = &dl_fn->nvenc_funcs;



    NVENCSTATUS nv_status;

    NV_ENC_CREATE_BITSTREAM_BUFFER allocOut = { 0 };

    allocOut.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;



    if (avctx->pix_fmt == AV_PIX_FMT_CUDA) {

        ctx->surfaces[idx].in_ref = av_frame_alloc();

        if (!ctx->surfaces[idx].in_ref)

            return AVERROR(ENOMEM);

    } else {

        NV_ENC_CREATE_INPUT_BUFFER allocSurf = { 0 };



        ctx->surfaces[idx].format = nvenc_map_buffer_format(ctx->data_pix_fmt);

        if (ctx->surfaces[idx].format == NV_ENC_BUFFER_FORMAT_UNDEFINED) {

            av_log(avctx, AV_LOG_FATAL, "Invalid input pixel format: %s\n",

                   av_get_pix_fmt_name(ctx->data_pix_fmt));

            return AVERROR(EINVAL);

        }



        allocSurf.version = NV_ENC_CREATE_INPUT_BUFFER_VER;

        allocSurf.width = (avctx->width + 31) & ~31;

        allocSurf.height = (avctx->height + 31) & ~31;

        allocSurf.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;

        allocSurf.bufferFmt = ctx->surfaces[idx].format;



        nv_status = p_nvenc->nvEncCreateInputBuffer(ctx->nvencoder, &allocSurf);

        if (nv_status != NV_ENC_SUCCESS) {

            return nvenc_print_error(avctx, nv_status, "CreateInputBuffer failed");

        }



        ctx->surfaces[idx].input_surface = allocSurf.inputBuffer;

        ctx->surfaces[idx].width = allocSurf.width;

        ctx->surfaces[idx].height = allocSurf.height;

    }



    ctx->surfaces[idx].lockCount = 0;



    /* 1MB is large enough to hold most output frames.

     * NVENC increases this automaticaly if it is not enough. */

    allocOut.size = 1024 * 1024;



    allocOut.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;



    nv_status = p_nvenc->nvEncCreateBitstreamBuffer(ctx->nvencoder, &allocOut);

    if (nv_status != NV_ENC_SUCCESS) {

        int err = nvenc_print_error(avctx, nv_status, "CreateBitstreamBuffer failed");

        if (avctx->pix_fmt != AV_PIX_FMT_CUDA)

            p_nvenc->nvEncDestroyInputBuffer(ctx->nvencoder, ctx->surfaces[idx].input_surface);

        av_frame_free(&ctx->surfaces[idx].in_ref);

        return err;

    }



    ctx->surfaces[idx].output_surface = allocOut.bitstreamBuffer;

    ctx->surfaces[idx].size = allocOut.size;



    return 0;

}
