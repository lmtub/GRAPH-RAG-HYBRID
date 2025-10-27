static void vda_decoder_callback(void *vda_hw_ctx,
                                 CFDictionaryRef user_info,
                                 OSStatus status,
                                 uint32_t infoFlags,
                                 CVImageBufferRef image_buffer)
{
    struct vda_context *vda_ctx = vda_hw_ctx;
    if (!image_buffer)
        return;
    if (vda_ctx->cv_pix_fmt_type != CVPixelBufferGetPixelFormatType(image_buffer))
        return;
    vda_ctx->cv_buffer = CVPixelBufferRetain(image_buffer);
}