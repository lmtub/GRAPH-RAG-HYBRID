static int xan_decode_frame(AVCodecContext *avctx,
                            void *data, int *data_size,
                            AVPacket *avpkt)
{
    const uint8_t *buf = avpkt->data;
    int ret, buf_size = avpkt->size;
    XanContext *s = avctx->priv_data;
    if (avctx->codec->id == CODEC_ID_XAN_WC3) {
        const uint8_t *buf_end = buf + buf_size;
        int tag = 0;
        while (buf_end - buf > 8 && tag != VGA__TAG) {
            unsigned *tmpptr;
            uint32_t new_pal;
            int size;
            int i;
            tag  = bytestream_get_le32(&buf);
            size = bytestream_get_be32(&buf);
            size = FFMIN(size, buf_end - buf);
            switch (tag) {
            case PALT_TAG:
                if (size < PALETTE_SIZE)
                if (s->palettes_count >= PALETTES_MAX)
                tmpptr = av_realloc(s->palettes, (s->palettes_count + 1) * AVPALETTE_SIZE);
                if (!tmpptr)
                    return AVERROR(ENOMEM);
                s->palettes = tmpptr;
                tmpptr += s->palettes_count * AVPALETTE_COUNT;
                for (i = 0; i < PALETTE_COUNT; i++) {
#if RUNTIME_GAMMA
                    int r = gamma_corr(*buf++);
                    int g = gamma_corr(*buf++);
                    int b = gamma_corr(*buf++);
#else
                    int r = gamma_lookup[*buf++];
                    int g = gamma_lookup[*buf++];
                    int b = gamma_lookup[*buf++];
#endif
                    *tmpptr++ = (r << 16) | (g << 8) | b;
                }
                s->palettes_count++;
                break;
            case SHOT_TAG:
                if (size < 4)
                new_pal = bytestream_get_le32(&buf);
                if (new_pal < s->palettes_count) {
                    s->cur_palette = new_pal;
                } else
                    av_log(avctx, AV_LOG_ERROR, "Invalid palette selected\n");
                break;
            case VGA__TAG:
                break;
            default:
                buf += size;
                break;
            }
        }
        buf_size = buf_end - buf;
    }
    if ((ret = avctx->get_buffer(avctx, &s->current_frame))) {
        av_log(s->avctx, AV_LOG_ERROR, "get_buffer() failed\n");
        return ret;
    }
    s->current_frame.reference = 3;
    if (!s->frame_size)
        s->frame_size = s->current_frame.linesize[0] * s->avctx->height;
    memcpy(s->current_frame.data[1], s->palettes + s->cur_palette * AVPALETTE_COUNT, AVPALETTE_SIZE);
    s->buf = buf;
    s->size = buf_size;
    if (xan_wc3_decode_frame(s) < 0)
    /* release the last frame if it is allocated */
    if (s->last_frame.data[0])
        avctx->release_buffer(avctx, &s->last_frame);
    *data_size = sizeof(AVFrame);
    *(AVFrame*)data = s->current_frame;
    /* shuffle frames */
    FFSWAP(AVFrame, s->current_frame, s->last_frame);
    /* always report that the buffer was completely consumed */
    return buf_size;
}