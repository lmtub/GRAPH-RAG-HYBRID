static int dxva2_mpeg2_end_frame(AVCodecContext *avctx)

{

    struct MpegEncContext *s = avctx->priv_data;

    struct dxva2_picture_context *ctx_pic =

        s->current_picture_ptr->hwaccel_picture_private;

    int ret;



    if (ctx_pic->slice_count <= 0 || ctx_pic->bitstream_size <= 0)

        return -1;

    ret = ff_dxva2_common_end_frame(avctx, &s->current_picture_ptr->f,

                                    &ctx_pic->pp, sizeof(ctx_pic->pp),

                                    &ctx_pic->qm, sizeof(ctx_pic->qm),

                                    commit_bitstream_and_slice_buffer);

    if (!ret)

        ff_mpeg_draw_horiz_band(s, 0, avctx->height);

    return ret;

}
