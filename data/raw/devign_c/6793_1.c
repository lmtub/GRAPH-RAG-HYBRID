static int mp_decode_frame(AVCodecContext *avctx,

                                 void *data, int *data_size,

                                 AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    MotionPixelsContext *mp = avctx->priv_data;

    GetBitContext gb;

    int i, count1, count2, sz;



    mp->frame.reference = 1;

    mp->frame.buffer_hints = FF_BUFFER_HINTS_VALID | FF_BUFFER_HINTS_PRESERVE | FF_BUFFER_HINTS_REUSABLE;

    if (avctx->reget_buffer(avctx, &mp->frame)) {

        av_log(avctx, AV_LOG_ERROR, "reget_buffer() failed\n");

        return -1;

    }



    /* le32 bitstream msb first */

    av_fast_malloc(&mp->bswapbuf, &mp->bswapbuf_size, buf_size + FF_INPUT_BUFFER_PADDING_SIZE);

    if (!mp->bswapbuf)

        return AVERROR(ENOMEM);

    mp->dsp.bswap_buf((uint32_t *)mp->bswapbuf, (const uint32_t *)buf, buf_size / 4);

    if (buf_size & 3)

        memcpy(mp->bswapbuf + (buf_size & ~3), buf + (buf_size & ~3), buf_size & 3);

    init_get_bits(&gb, mp->bswapbuf, buf_size * 8);



    memset(mp->changes_map, 0, avctx->width * avctx->height);

    for (i = !(avctx->extradata[1] & 2); i < 2; ++i) {

        count1 = get_bits(&gb, 12);

        count2 = get_bits(&gb, 12);

        mp_read_changes_map(mp, &gb, count1, 8, i);

        mp_read_changes_map(mp, &gb, count2, 4, i);

    }



    mp->codes_count = get_bits(&gb, 4);

    if (mp->codes_count == 0)

        goto end;



    if (mp->changes_map[0] == 0) {

        *(uint16_t *)mp->frame.data[0] = get_bits(&gb, 15);

        mp->changes_map[0] = 1;

    }

    mp_read_codes_table(mp, &gb);



    sz = get_bits(&gb, 18);

    if (avctx->extradata[0] != 5)

        sz += get_bits(&gb, 18);

    if (sz == 0)

        goto end;



    init_vlc(&mp->vlc, mp->max_codes_bits, mp->codes_count, &mp->codes[0].size, sizeof(HuffCode), 1, &mp->codes[0].code, sizeof(HuffCode), 4, 0);

    mp_decode_frame_helper(mp, &gb);

    free_vlc(&mp->vlc);



end:

    *data_size = sizeof(AVFrame);

    *(AVFrame *)data = mp->frame;

    return buf_size;

}
