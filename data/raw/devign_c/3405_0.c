static int dxva2_hevc_start_frame(AVCodecContext *avctx,

                                  av_unused const uint8_t *buffer,

                                  av_unused uint32_t size)

{

    const HEVCContext *h = avctx->priv_data;

    AVDXVAContext *ctx = avctx->hwaccel_context;

    struct hevc_dxva2_picture_context *ctx_pic = h->ref->hwaccel_picture_private;



    if (!DXVA_CONTEXT_VALID(avctx, ctx))

        return -1;

    av_assert0(ctx_pic);



    /* Fill up DXVA_PicParams_HEVC */

    fill_picture_parameters(avctx, ctx, h, &ctx_pic->pp);



    /* Fill up DXVA_Qmatrix_HEVC */

    fill_scaling_lists(ctx, h, &ctx_pic->qm);



    ctx_pic->slice_count    = 0;

    ctx_pic->bitstream_size = 0;

    ctx_pic->bitstream      = NULL;

    return 0;

}
