static int dxtory_decode_v2_444(AVCodecContext *avctx, AVFrame *pic,

                                const uint8_t *src, int src_size)

{

    GetByteContext gb;

    GetBitContext  gb2;

    int nslices, slice, slice_height;

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

        return AVERROR_PATCHWELCOME;

    }



    slice_height = avctx->height / nslices;



    avctx->pix_fmt = AV_PIX_FMT_YUV444P;

    if ((ret = ff_get_buffer(avctx, pic, 0)) < 0)

        return ret;



    Y = pic->data[0];

    U = pic->data[1];

    V = pic->data[2];



    for (slice = 0; slice < nslices; slice++) {

        slice_size = bytestream2_get_le32(&gb);



        ret = check_slice_size(avctx, src, src_size, slice_size, off);

        if (ret < 0)

            return ret;



        init_get_bits(&gb2, src + off + 16, (slice_size - 16) * 8);

        dx2_decode_slice_444(&gb2, avctx->width, slice_height, Y, U, V,

                             pic->linesize[0], pic->linesize[1],

                             pic->linesize[2]);



        Y += pic->linesize[0] * slice_height;

        U += pic->linesize[1] * slice_height;

        V += pic->linesize[2] * slice_height;

        off += slice_size;

    }



    return 0;

}
