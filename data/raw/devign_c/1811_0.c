static void cuvid_flush(AVCodecContext *avctx)

{

    CuvidContext *ctx = avctx->priv_data;

    AVHWDeviceContext *device_ctx = (AVHWDeviceContext*)ctx->hwdevice->data;

    AVCUDADeviceContext *device_hwctx = device_ctx->hwctx;

    CUcontext dummy, cuda_ctx = device_hwctx->cuda_ctx;

    CUVIDSOURCEDATAPACKET seq_pkt = { 0 };

    int ret;



    ctx->ever_flushed = 1;



    ret = CHECK_CU(cuCtxPushCurrent(cuda_ctx));

    if (ret < 0)

        goto error;



    av_fifo_freep(&ctx->frame_queue);



    ctx->frame_queue = av_fifo_alloc(MAX_FRAME_COUNT * sizeof(CuvidParsedFrame));

    if (!ctx->frame_queue) {

        av_log(avctx, AV_LOG_ERROR, "Failed to recreate frame queue on flush\n");

        return;

    }



    if (ctx->cudecoder) {

        cuvidDestroyDecoder(ctx->cudecoder);

        ctx->cudecoder = NULL;

    }



    if (ctx->cuparser) {

        cuvidDestroyVideoParser(ctx->cuparser);

        ctx->cuparser = NULL;

    }



    ret = CHECK_CU(cuvidCreateVideoParser(&ctx->cuparser, &ctx->cuparseinfo));

    if (ret < 0)

        goto error;



    seq_pkt.payload = ctx->cuparse_ext.raw_seqhdr_data;

    seq_pkt.payload_size = ctx->cuparse_ext.format.seqhdr_data_length;



    if (seq_pkt.payload && seq_pkt.payload_size) {

        ret = CHECK_CU(cuvidParseVideoData(ctx->cuparser, &seq_pkt));

        if (ret < 0)

            goto error;

    }



    ret = CHECK_CU(cuCtxPopCurrent(&dummy));

    if (ret < 0)

        goto error;



    ctx->prev_pts = INT64_MIN;

    ctx->decoder_flushing = 0;



    return;

 error:

    av_log(avctx, AV_LOG_ERROR, "CUDA reinit on flush failed\n");

}
