static int dxtory_decode_v2(AVCodecContext *avctx, AVFrame *pic,

                            const uint8_t *src, int src_size)

{

    GetByteContext gb;

    GetBitContext  gb2;

    int nslices, slice, slice_height, ref_slice_height;

    int cur_y, next_y;

    uint32_t off, slice_size;

    uint8_t *Y, *U, *V;

    int ret;



    bytestream2_init(&gb, src, src_size);

    nslices = bytestream2_get_le16(&gb);

    off = FFALIGN(nslices * 4 + 2, 16);

    if (src_size < off) {

        av_log(avctx, AV_LOG_ERROR, "no slice data\n");

        return AVERROR_INVALIDDATA;

    }



    if (!nslices || avctx->height % nslices) {

        avpriv_request_sample(avctx, "%d slices for %dx%d", nslices,

                              avctx->width, avctx->height);

        return AVERROR(ENOSYS);

    }



    ref_slice_height = avctx->height / nslices;

    if ((avctx->width & 1) || (avctx->height & 1)) {

        avpriv_request_sample(avctx, "Frame dimensions %dx%d",

                              avctx->width, avctx->height);

    }



    avctx->pix_fmt = AV_PIX_FMT_YUV420P;

    if ((ret = ff_get_buffer(avctx, pic, 0)) < 0)

        return ret;



    Y = pic->data[0];

    U = pic->data[1];

    V = pic->data[2];



    cur_y  = 0;

    next_y = ref_slice_height;

    for (slice = 0; slice < nslices; slice++) {

        slice_size   = bytestream2_get_le32(&gb);

        slice_height = (next_y & ~1) - (cur_y & ~1);

        if (slice_size > src_size - off) {

            av_log(avctx, AV_LOG_ERROR,

                   "invalid slice size %d (only %d bytes left)\n",

                   slice_size, src_size - off);

            return AVERROR_INVALIDDATA;

        }

        if (slice_size <= 16) {

            av_log(avctx, AV_LOG_ERROR, "invalid slice size %d\n", slice_size);

            return AVERROR_INVALIDDATA;

        }



        if (AV_RL32(src + off) != slice_size - 16) {

            av_log(avctx, AV_LOG_ERROR,

                   "Slice sizes mismatch: got %d instead of %d\n",

                   AV_RL32(src + off), slice_size - 16);

        }

        init_get_bits(&gb2, src + off + 16, (slice_size - 16) * 8);

        dx2_decode_slice(&gb2, avctx->width, slice_height, Y, U, V,

                         pic->linesize[0], pic->linesize[1], pic->linesize[2]);



        Y += pic->linesize[0] *  slice_height;

        U += pic->linesize[1] * (slice_height >> 1);

        V += pic->linesize[2] * (slice_height >> 1);

        off += slice_size;

        cur_y   = next_y;

        next_y += ref_slice_height;

    }



    return 0;

}
