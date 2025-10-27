static int dxva2_vc1_end_frame(AVCodecContext *avctx)

{

    VC1Context *v = avctx->priv_data;

    struct dxva2_picture_context *ctx_pic = v->s.current_picture_ptr->hwaccel_picture_private;

    int ret;



    if (ctx_pic->bitstream_size <= 0)

        return -1;



    ret = ff_dxva2_common_end_frame(avctx, &v->s.current_picture_ptr->f,

                                    &ctx_pic->pp, sizeof(ctx_pic->pp),

                                    NULL, 0,

                                    commit_bitstream_and_slice_buffer);

    if (!ret)

        ff_mpeg_draw_horiz_band(&v->s, 0, avctx->height);

    return ret;

}
