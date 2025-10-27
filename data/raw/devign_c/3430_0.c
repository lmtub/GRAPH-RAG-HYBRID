static int nvdec_vc1_decode_slice(AVCodecContext *avctx, const uint8_t *buffer, uint32_t size)

{

    NVDECContext *ctx = avctx->internal->hwaccel_priv_data;

    void *tmp;



    tmp = av_fast_realloc(ctx->slice_offsets, &ctx->slice_offsets_allocated,

                          (ctx->nb_slices + 1) * sizeof(*ctx->slice_offsets));

    if (!tmp)

        return AVERROR(ENOMEM);

    ctx->slice_offsets = tmp;



    if (!ctx->bitstream)

        ctx->bitstream = (uint8_t*)buffer;



    ctx->slice_offsets[ctx->nb_slices] = buffer - ctx->bitstream;

    ctx->bitstream_len += size;

    ctx->nb_slices++;



    return 0;

}
