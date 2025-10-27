static int cllc_decode_frame(AVCodecContext *avctx, void *data,

                             int *got_picture_ptr, AVPacket *avpkt)

{

    CLLCContext *ctx = avctx->priv_data;

    AVFrame *pic = avctx->coded_frame;

    uint8_t *src = avpkt->data;

    uint8_t *swapped_buf_new;

    uint32_t info_tag, info_offset;

    GetBitContext gb;

    int coding_type, ret;



    if (pic->data[0])

        avctx->release_buffer(avctx, pic);



    pic->reference = 0;



    /* Make sure our bswap16'd buffer is big enough */

    swapped_buf_new = av_fast_realloc(ctx->swapped_buf,

                                      &ctx->swapped_buf_size, avpkt->size);

    if (!swapped_buf_new) {

        av_log(avctx, AV_LOG_ERROR, "Could not realloc swapped buffer.\n");

        return AVERROR(ENOMEM);

    }

    ctx->swapped_buf = swapped_buf_new;



    /* Skip the INFO header if present */

    info_offset = 0;

    info_tag    = AV_RL32(src);

    if (info_tag == MKTAG('I', 'N', 'F', 'O')) {

        info_offset = AV_RL32(src + 4);

        if (info_offset > UINT32_MAX - 8 || info_offset + 8 > avpkt->size) {

            av_log(avctx, AV_LOG_ERROR,

                   "Invalid INFO header offset: 0x%08X is too large.\n",

                   info_offset);

            return AVERROR_INVALIDDATA;

        }



        info_offset += 8;

        src         += info_offset;



        av_log(avctx, AV_LOG_DEBUG, "Skipping INFO chunk.\n");

    }



    /* bswap16 the buffer since CLLC's bitreader works in 16-bit words */

    ctx->dsp.bswap16_buf((uint16_t *) ctx->swapped_buf, (uint16_t *) src,

                         (avpkt->size - info_offset) / 2);



    init_get_bits(&gb, ctx->swapped_buf, (avpkt->size - info_offset) * 8);



    /*

     * Read in coding type. The types are as follows:

     *

     * 0 - YUY2

     * 1 - BGR24 (Triples)

     * 2 - BGR24 (Quads)

     * 3 - BGRA

     */

    coding_type = (AV_RL32(src) >> 8) & 0xFF;

    av_log(avctx, AV_LOG_DEBUG, "Frame coding type: %d\n", coding_type);



    switch (coding_type) {

    case 1:

    case 2:

        avctx->pix_fmt             = PIX_FMT_RGB24;

        avctx->bits_per_raw_sample = 8;



        ret = avctx->get_buffer(avctx, pic);

        if (ret < 0) {

            av_log(avctx, AV_LOG_ERROR, "Could not allocate buffer.\n");

            return ret;

        }



        ret = decode_rgb24_frame(ctx, &gb, pic);

        if (ret < 0)

            return ret;



        break;

    case 3:

        avctx->pix_fmt             = PIX_FMT_ARGB;

        avctx->bits_per_raw_sample = 8;



        ret = avctx->get_buffer(avctx, pic);

        if (ret < 0) {

            av_log(avctx, AV_LOG_ERROR, "Could not allocate buffer.\n");

            return ret;

        }



        ret = decode_argb_frame(ctx, &gb, pic);

        if (ret < 0)

            return ret;



        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "Unknown coding type: %d.\n", coding_type);

        return AVERROR_INVALIDDATA;

    }



    pic->key_frame = 1;

    pic->pict_type = AV_PICTURE_TYPE_I;



    *got_picture_ptr = 1;

    *(AVFrame *)data = *pic;



    return avpkt->size;

}
