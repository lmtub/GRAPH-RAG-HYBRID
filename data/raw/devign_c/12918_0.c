static void dxva2_uninit(AVCodecContext *s)

{

    InputStream  *ist = s->opaque;

    DXVA2Context *ctx = ist->hwaccel_ctx;



    ist->hwaccel_uninit        = NULL;

    ist->hwaccel_get_buffer    = NULL;

    ist->hwaccel_retrieve_data = NULL;



    if (ctx->decoder_service)

        IDirectXVideoDecoderService_Release(ctx->decoder_service);



    av_buffer_unref(&ctx->hw_frames_ctx);

    av_buffer_unref(&ctx->hw_device_ctx);



    av_frame_free(&ctx->tmp_frame);



    av_freep(&ist->hwaccel_ctx);

    av_freep(&s->hwaccel_context);

}
