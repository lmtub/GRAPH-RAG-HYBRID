static int decode_frame(AVCodecContext *avctx, void *data, int *got_frame,
                        AVPacket *avpkt)
{
    AVFrame *frame = data;
    const uint8_t *buf = avpkt->data;
    const uint8_t *buf_end = buf + avpkt->size;
    KgvContext * const c = avctx->priv_data;
    int offsets[8];
    uint8_t *out, *prev;
    int outcnt = 0, maxcnt;
    int w, h, i, res;
    if (avpkt->size < 2)
    w = (buf[0] + 1) * 8;
    h = (buf[1] + 1) * 8;
    buf += 2;
    if (w != avctx->width || h != avctx->height) {
        av_freep(&c->frame_buffer);
        av_freep(&c->last_frame_buffer);
        if ((res = ff_set_dimensions(avctx, w, h)) < 0)
            return res;
    }
    if (!c->frame_buffer) {
        c->frame_buffer      = av_mallocz(avctx->width * avctx->height * 2);
        c->last_frame_buffer = av_mallocz(avctx->width * avctx->height * 2);
        if (!c->frame_buffer || !c->last_frame_buffer) {
            decode_flush(avctx);
            return AVERROR(ENOMEM);
        }
    }
    maxcnt = w * h;
    if ((res = ff_get_buffer(avctx, frame, 0)) < 0)
        return res;
    out  = (uint8_t*)c->frame_buffer;
    prev = (uint8_t*)c->last_frame_buffer;
    for (i = 0; i < 8; i++)
        offsets[i] = -1;
    while (outcnt < maxcnt && buf_end - 2 >= buf) {
        int code = AV_RL16(buf);
        buf += 2;
        if (!(code & 0x8000)) {
            AV_WN16A(&out[2 * outcnt], code); // rgb555 pixel coded directly
            outcnt++;
        } else {
            int count;
            if ((code & 0x6000) == 0x6000) {
                // copy from previous frame
                int oidx = (code >> 10) & 7;
                int start;
                count = (code & 0x3FF) + 3;
                if (offsets[oidx] < 0) {
                    if (buf_end - 3 < buf)
                        break;
                    offsets[oidx] = AV_RL24(buf);
                    buf += 3;
                }
                start = (outcnt + offsets[oidx]) % maxcnt;
                if (maxcnt - start < count || maxcnt - outcnt < count)
                    break;
                if (!prev) {
                    av_log(avctx, AV_LOG_ERROR,
                           "Frame reference does not exist\n");
                    break;
                }
                memcpy(out + 2 * outcnt, prev + 2 * start, 2 * count);
            } else {
                // copy from earlier in this frame
                int offset = (code & 0x1FFF) + 1;
                if (!(code & 0x6000)) {
                    count = 2;
                } else if ((code & 0x6000) == 0x2000) {
                    count = 3;
                } else {
                    if (buf_end - 1 < buf)
                        break;
                    count = 4 + *buf++;
                }
                if (outcnt < offset || maxcnt - outcnt < count)
                    break;
                av_memcpy_backptr(out + 2 * outcnt, 2 * offset, 2 * count);
            }
            outcnt += count;
        }
    }
    if (outcnt - maxcnt)
        av_log(avctx, AV_LOG_DEBUG, "frame finished with %d diff\n", outcnt - maxcnt);
    av_image_copy_plane(frame->data[0], frame->linesize[0],
                        (const uint8_t*)c->frame_buffer,  avctx->width * 2,
                        avctx->width * 2, avctx->height);
    FFSWAP(uint16_t *, c->frame_buffer, c->last_frame_buffer);
    *got_frame = 1;
    return avpkt->size;
}