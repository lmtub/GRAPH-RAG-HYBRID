static int dxva2_vc1_start_frame(AVCodecContext *avctx,

                                 av_unused const uint8_t *buffer,

                                 av_unused uint32_t size)

{

    const VC1Context *v = avctx->priv_data;

    AVDXVAContext *ctx = avctx->hwaccel_context;

    struct dxva2_picture_context *ctx_pic = v->s.current_picture_ptr->hwaccel_picture_private;



    if (!DXVA_CONTEXT_VALID(avctx, ctx))

        return -1;

    assert(ctx_pic);



    fill_picture_parameters(avctx, ctx, v, &ctx_pic->pp);



    ctx_pic->bitstream_size = 0;

    ctx_pic->bitstream      = NULL;

    return 0;

}
