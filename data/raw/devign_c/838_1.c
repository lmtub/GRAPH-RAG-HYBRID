static int cllc_decode_frame(AVCodecContext *avctx, void *data,
                             int *got_picture_ptr, AVPacket *avpkt)
{
    CLLCContext *ctx = avctx->priv_data;
    AVFrame *pic = data;
    ThreadFrame frame = { .f = data };
    uint8_t *src = avpkt->data;
    uint32_t info_tag, info_offset;
    int data_size;
    GetBitContext gb;
    int coding_type, ret;
    if (avpkt->size < 4 + 4) {
        av_log(avctx, AV_LOG_ERROR, "Frame is too small %d.\n", avpkt->size);
    }
    info_offset = 0;
    info_tag    = AV_RL32(src);
    if (info_tag == MKTAG('I', 'N', 'F', 'O')) {
        info_offset = AV_RL32(src + 4);
        if (info_offset > UINT32_MAX - 8 || info_offset + 8 > avpkt->size) {
            av_log(avctx, AV_LOG_ERROR,
                   "Invalid INFO header offset: 0x%08"PRIX32" is too large.\n",
                   info_offset);
        }
        ff_canopus_parse_info_tag(avctx, src + 8, info_offset);
        info_offset += 8;
        src         += info_offset;
    }
    data_size = (avpkt->size - info_offset) & ~1;
    /* Make sure our bswap16'd buffer is big enough */
    av_fast_padded_malloc(&ctx->swapped_buf,
                          &ctx->swapped_buf_size, data_size);
    if (!ctx->swapped_buf) {
        av_log(avctx, AV_LOG_ERROR, "Could not allocate swapped buffer.\n");
        return AVERROR(ENOMEM);
    }
    /* bswap16 the buffer since CLLC's bitreader works in 16-bit words */
    ctx->bdsp.bswap16_buf((uint16_t *) ctx->swapped_buf, (uint16_t *) src,
                          data_size / 2);
    if ((ret = init_get_bits8(&gb, ctx->swapped_buf, data_size)) < 0)
        return ret;
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
    case 0:
        avctx->pix_fmt             = AV_PIX_FMT_YUV422P;
        avctx->bits_per_raw_sample = 8;
        if ((ret = ff_thread_get_buffer(avctx, &frame, 0)) < 0)
            return ret;
        ret = decode_yuv_frame(ctx, &gb, pic);
        if (ret < 0)
            return ret;
        break;
    case 1:
    case 2:
        avctx->pix_fmt             = AV_PIX_FMT_RGB24;
        avctx->bits_per_raw_sample = 8;
        if ((ret = ff_thread_get_buffer(avctx, &frame, 0)) < 0)
            return ret;
        ret = decode_rgb24_frame(ctx, &gb, pic);
        if (ret < 0)
            return ret;
        break;
    case 3:
        avctx->pix_fmt             = AV_PIX_FMT_ARGB;
        avctx->bits_per_raw_sample = 8;
        if ((ret = ff_thread_get_buffer(avctx, &frame, 0)) < 0)
            return ret;
        ret = decode_argb_frame(ctx, &gb, pic);
        if (ret < 0)
            return ret;
        break;
    default:
        av_log(avctx, AV_LOG_ERROR, "Unknown coding type: %d.\n", coding_type);
    }
    pic->key_frame = 1;
    pic->pict_type = AV_PICTURE_TYPE_I;
    *got_picture_ptr = 1;
    return avpkt->size;
}