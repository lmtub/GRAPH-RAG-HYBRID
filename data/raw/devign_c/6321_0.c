static int dxva2_mpeg2_start_frame(AVCodecContext *avctx,

                                   av_unused const uint8_t *buffer,

                                   av_unused uint32_t size)

{

    const struct MpegEncContext *s = avctx->priv_data;

    AVDXVAContext *ctx = avctx->hwaccel_context;

    struct dxva2_picture_context *ctx_pic =

        s->current_picture_ptr->hwaccel_picture_private;



    if (!DXVA_CONTEXT_VALID(avctx, ctx))

        return -1;

    assert(ctx_pic);



    fill_picture_parameters(avctx, ctx, s, &ctx_pic->pp);

    fill_quantization_matrices(avctx, ctx, s, &ctx_pic->qm);



    ctx_pic->slice_count    = 0;

    ctx_pic->bitstream_size = 0;

    ctx_pic->bitstream      = NULL;

    return 0;

}
