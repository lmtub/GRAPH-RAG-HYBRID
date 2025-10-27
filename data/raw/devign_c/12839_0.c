static int decode_picture_header(AVCodecContext *avctx, const uint8_t *buf, const int buf_size)

{

    ProresContext *ctx = avctx->priv_data;

    int i, hdr_size, slice_count;

    unsigned pic_data_size;

    int log2_slice_mb_width, log2_slice_mb_height;

    int slice_mb_count, mb_x, mb_y;

    const uint8_t *data_ptr, *index_ptr;



    hdr_size = buf[0] >> 3;

    if (hdr_size < 8 || hdr_size > buf_size) {

        av_log(avctx, AV_LOG_ERROR, "error, wrong picture header size\n");

        return -1;

    }



    pic_data_size = AV_RB32(buf + 1);

    if (pic_data_size > buf_size) {

        av_log(avctx, AV_LOG_ERROR, "error, wrong picture data size\n");

        return -1;

    }



    log2_slice_mb_width  = buf[7] >> 4;

    log2_slice_mb_height = buf[7] & 0xF;

    if (log2_slice_mb_width > 3 || log2_slice_mb_height) {

        av_log(avctx, AV_LOG_ERROR, "unsupported slice resolution: %dx%d\n",

               1 << log2_slice_mb_width, 1 << log2_slice_mb_height);

        return -1;

    }



    ctx->mb_width  = (avctx->width  + 15) >> 4;

    ctx->mb_height = (avctx->height + 15) >> 4;



    slice_count = AV_RB16(buf + 5);



    if (ctx->slice_count != slice_count || !ctx->slices) {

        av_freep(&ctx->slices);

        ctx->slices = av_mallocz(slice_count * sizeof(*ctx->slices));

        if (!ctx->slices)

            return AVERROR(ENOMEM);

        ctx->slice_count = slice_count;

    }



    if (!slice_count)

        return AVERROR(EINVAL);



    if (hdr_size + slice_count*2 > buf_size) {

        av_log(avctx, AV_LOG_ERROR, "error, wrong slice count\n");

        return -1;

    }



    // parse slice information

    index_ptr = buf + hdr_size;

    data_ptr  = index_ptr + slice_count*2;



    slice_mb_count = 1 << log2_slice_mb_width;

    mb_x = 0;

    mb_y = 0;



    for (i = 0; i < slice_count; i++) {

        SliceContext *slice = &ctx->slices[i];



        slice->data = data_ptr;

        data_ptr += AV_RB16(index_ptr + i*2);



        while (ctx->mb_width - mb_x < slice_mb_count)

            slice_mb_count >>= 1;



        slice->mb_x = mb_x;

        slice->mb_y = mb_y;

        slice->mb_count = slice_mb_count;

        slice->data_size = data_ptr - slice->data;



        if (slice->data_size < 6) {

            av_log(avctx, AV_LOG_ERROR, "error, wrong slice data size\n");

            return -1;

        }



        mb_x += slice_mb_count;

        if (mb_x == ctx->mb_width) {

            slice_mb_count = 1 << log2_slice_mb_width;

            mb_x = 0;

            mb_y++;

        }

        if (data_ptr > buf + buf_size) {

            av_log(avctx, AV_LOG_ERROR, "error, slice out of bounds\n");

            return -1;

        }

    }



    return pic_data_size;

}
