static int vda_h264_decode_slice(AVCodecContext *avctx,

                                 const uint8_t *buffer,

                                 uint32_t size)

{

    VDAContext *vda                     = avctx->internal->hwaccel_priv_data;

    struct vda_context *vda_ctx         = avctx->hwaccel_context;

    void *tmp;



    if (!vda_ctx->decoder)

        return -1;



    tmp = av_fast_realloc(vda->bitstream,

                          &vda->allocated_size,

                          vda->bitstream_size + size + 4);

    if (!tmp)

        return AVERROR(ENOMEM);



    vda->bitstream = tmp;



    AV_WB32(vda->bitstream + vda->bitstream_size, size);

    memcpy(vda->bitstream + vda->bitstream_size + 4, buffer, size);



    vda->bitstream_size += size + 4;



    return 0;

}
