static int set_hwframe_ctx(AVCodecContext *ctx, AVBufferRef *hw_device_ctx)

{

    AVBufferRef *hw_frames_ref;

    AVHWFramesContext *frames_ctx = NULL;

    int err = 0;



    if (!(hw_frames_ref = av_hwframe_ctx_alloc(hw_device_ctx))) {

        fprintf(stderr, "Failed to create VAAPI frame context.\n");

        return -1;

    }

    frames_ctx = (AVHWFramesContext *)(hw_frames_ref->data);

    frames_ctx->format    = AV_PIX_FMT_VAAPI;

    frames_ctx->sw_format = AV_PIX_FMT_NV12;

    frames_ctx->width     = width;

    frames_ctx->height    = height;

    frames_ctx->initial_pool_size = 20;

    if ((err = av_hwframe_ctx_init(hw_frames_ref)) < 0) {

        fprintf(stderr, "Failed to initialize VAAPI frame context."

                "Error code: %s\n",av_err2str(err));


        return err;

    }

    ctx->hw_frames_ctx = av_buffer_ref(hw_frames_ref);

    if (!ctx->hw_frames_ctx)

        err = AVERROR(ENOMEM);




    return err;

}