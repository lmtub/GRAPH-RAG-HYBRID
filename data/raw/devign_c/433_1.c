static int decode_frame(AVCodecContext *avctx, void *data, int *data_size,
                        AVPacket *avpkt) {
    const uint8_t *buf = avpkt->data;
    int buf_size = avpkt->size;
    AVSubtitle *sub = data;
    const uint8_t *buf_end = buf + buf_size;
    uint8_t *bitmap;
    int w, h, x, y, i, ret;
    int64_t packet_time = 0;
    GetBitContext gb;
    int has_alpha = avctx->codec_tag == MKTAG('D','X','S','A');
    // check that at least header fits
    if (buf_size < 27 + 7 * 2 + 4 * (3 + has_alpha)) {
        av_log(avctx, AV_LOG_ERROR, "coded frame size %d too small\n", buf_size);
        return -1;
    }
    // read start and end time
    if (buf[0] != '[' || buf[13] != '-' || buf[26] != ']') {
        av_log(avctx, AV_LOG_ERROR, "invalid time code\n");
        return -1;
    }
    if (avpkt->pts != AV_NOPTS_VALUE)
        packet_time = av_rescale_q(avpkt->pts, AV_TIME_BASE_Q, (AVRational){1, 1000});
    sub->start_display_time = parse_timecode(buf +  1, packet_time);
    sub->end_display_time   = parse_timecode(buf + 14, packet_time);
    buf += 27;
    // read header
    w = bytestream_get_le16(&buf);
    h = bytestream_get_le16(&buf);
    if (av_image_check_size(w, h, 0, avctx) < 0)
        return -1;
    x = bytestream_get_le16(&buf);
    y = bytestream_get_le16(&buf);
    // skip bottom right position, it gives no new information
    bytestream_get_le16(&buf);
    bytestream_get_le16(&buf);
    // The following value is supposed to indicate the start offset
    // (relative to the palette) of the data for the second field,
    // however there are files in which it has a bogus value and thus
    // we just ignore it
    bytestream_get_le16(&buf);
    // allocate sub and set values
    sub->rects =  av_mallocz(sizeof(*sub->rects));
    if (!sub->rects)
        return AVERROR(ENOMEM);
    sub->rects[0] = av_mallocz(sizeof(*sub->rects[0]));
    if (!sub->rects[0]) {
        av_freep(&sub->rects);
        return AVERROR(ENOMEM);
    }
    sub->rects[0]->x = x; sub->rects[0]->y = y;
    sub->rects[0]->w = w; sub->rects[0]->h = h;
    sub->rects[0]->type = SUBTITLE_BITMAP;
    sub->rects[0]->linesize[0] = w;
    sub->rects[0]->data[0] = av_malloc(w * h);
    sub->rects[0]->nb_colors = 4;
    sub->rects[0]->data[1] = av_mallocz(AVPALETTE_SIZE);
    if (!sub->rects[0]->data[0] || !sub->rects[0]->data[1]) {
        av_freep(&sub->rects[0]->data[1]);
        av_freep(&sub->rects[0]->data[0]);
        av_freep(&sub->rects[0]);
        av_freep(&sub->rects);
        return AVERROR(ENOMEM);
    }
    sub->num_rects = 1;
    // read palette
    for (i = 0; i < sub->rects[0]->nb_colors; i++)
        ((uint32_t*)sub->rects[0]->data[1])[i] = bytestream_get_be24(&buf);
    if (!has_alpha) {
        // make all except background (first entry) non-transparent
        for (i = 1; i < sub->rects[0]->nb_colors; i++)
            ((uint32_t *)sub->rects[0]->data[1])[i] |= 0xff000000;
    } else {
        for (i = 0; i < sub->rects[0]->nb_colors; i++)
            ((uint32_t *)sub->rects[0]->data[1])[i] |= *buf++ << 24;
    }
#if FF_API_AVPICTURE
FF_DISABLE_DEPRECATION_WARNINGS
{
    AVSubtitleRect *rect;
    int j;
    rect = sub->rects[0];
    for (j = 0; j < 4; j++) {
        rect->pict.data[j] = rect->data[j];
        rect->pict.linesize[j] = rect->linesize[j];
    }
}
FF_ENABLE_DEPRECATION_WARNINGS
#endif
    // process RLE-compressed data
    if ((ret = init_get_bits8(&gb, buf, buf_end - buf)) < 0)
        return ret;
    bitmap = sub->rects[0]->data[0];
    for (y = 0; y < h; y++) {
        // interlaced: do odd lines
        if (y == (h + 1) / 2) bitmap = sub->rects[0]->data[0] + w;
        for (x = 0; x < w; ) {
            int log2 = ff_log2_tab[show_bits(&gb, 8)];
            int run = get_bits(&gb, 14 - 4 * (log2 >> 1));
            int color = get_bits(&gb, 2);
            run = FFMIN(run, w - x);
            // run length 0 means till end of row
            if (!run) run = w - x;
            memset(bitmap, color, run);
            bitmap += run;
            x += run;
        }
        // interlaced, skip every second line
        bitmap += w;
        align_get_bits(&gb);
    }
    *data_size = 1;
    return buf_size;
}