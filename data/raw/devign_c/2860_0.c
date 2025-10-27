static int dxva2_vp9_start_frame(AVCodecContext *avctx,

                                 av_unused const uint8_t *buffer,

                                 av_unused uint32_t size)

{

    const VP9SharedContext *h = avctx->priv_data;

    AVDXVAContext *ctx = avctx->hwaccel_context;

    struct vp9_dxva2_picture_context *ctx_pic = h->frames[CUR_FRAME].hwaccel_picture_private;



    if (DXVA_CONTEXT_DECODER(avctx, ctx) == NULL ||

        DXVA_CONTEXT_CFG(avctx, ctx) == NULL ||

        DXVA_CONTEXT_COUNT(avctx, ctx) <= 0)

        return -1;

    av_assert0(ctx_pic);



    /* Fill up DXVA_PicParams_VP9 */

    if (fill_picture_parameters(avctx, ctx, h, &ctx_pic->pp) < 0)

        return -1;



    ctx_pic->bitstream_size = 0;

    ctx_pic->bitstream      = NULL;

    return 0;

}
