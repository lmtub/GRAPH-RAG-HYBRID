static int dxva2_h264_decode_slice(AVCodecContext *avctx,

                                   const uint8_t *buffer,

                                   uint32_t size)

{

    const H264Context *h = avctx->priv_data;

    struct dxva_context *ctx = avctx->hwaccel_context;

    const Picture *current_picture = h->cur_pic_ptr;

    struct dxva2_picture_context *ctx_pic = current_picture->hwaccel_picture_private;

    unsigned position;



    if (ctx_pic->slice_count >= MAX_SLICES)

        return -1;



    if (!ctx_pic->bitstream)

        ctx_pic->bitstream = buffer;

    ctx_pic->bitstream_size += size;



    position = buffer - ctx_pic->bitstream;

    if (is_slice_short(ctx))

        fill_slice_short(&ctx_pic->slice_short[ctx_pic->slice_count],

                         position, size);

    else

        fill_slice_long(avctx, &ctx_pic->slice_long[ctx_pic->slice_count],

                        position, size);

    ctx_pic->slice_count++;



    if (h->slice_type != AV_PICTURE_TYPE_I && h->slice_type != AV_PICTURE_TYPE_SI)

        ctx_pic->pp.wBitFields &= ~(1 << 15); /* Set IntraPicFlag to 0 */

    return 0;

}
