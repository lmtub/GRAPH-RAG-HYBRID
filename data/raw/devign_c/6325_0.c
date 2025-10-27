static int dxva2_h264_start_frame(AVCodecContext *avctx,

                                  av_unused const uint8_t *buffer,

                                  av_unused uint32_t size)

{

    const H264Context *h = avctx->priv_data;

    AVDXVAContext *ctx = avctx->hwaccel_context;

    struct dxva2_picture_context *ctx_pic = h->cur_pic_ptr->hwaccel_picture_private;



    if (DXVA_CONTEXT_DECODER(avctx, ctx) == NULL ||

        DXVA_CONTEXT_CFG(avctx, ctx) == NULL ||

        DXVA_CONTEXT_COUNT(avctx, ctx) <= 0)

        return -1;

    assert(ctx_pic);



    /* Fill up DXVA_PicParams_H264 */

    fill_picture_parameters(avctx, ctx, h, &ctx_pic->pp);



    /* Fill up DXVA_Qmatrix_H264 */

    fill_scaling_lists(avctx, ctx, h, &ctx_pic->qm);



    ctx_pic->slice_count    = 0;

    ctx_pic->bitstream_size = 0;

    ctx_pic->bitstream      = NULL;

    return 0;

}
