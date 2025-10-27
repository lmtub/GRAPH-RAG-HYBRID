static int paf_vid_decode(AVCodecContext *avctx, void *data,

                          int *data_size, AVPacket *pkt)

{

    PAFVideoDecContext *c = avctx->priv_data;

    uint8_t code, *dst, *src, *end;

    int i, frame, ret;



    c->pic.reference = 3;

    if ((ret = avctx->reget_buffer(avctx, &c->pic)) < 0)

        return ret;



    bytestream2_init(&c->gb, pkt->data, pkt->size);



    code = bytestream2_get_byte(&c->gb);

    if (code & 0x20) {

        for (i = 0; i < 4; i++)

            memset(c->frame[i], 0, c->frame_size);



        memset(c->pic.data[1], 0, AVPALETTE_SIZE);

        c->current_frame = 0;

        c->pic.key_frame = 1;

        c->pic.pict_type = AV_PICTURE_TYPE_I;

    } else {

        c->pic.key_frame = 0;

        c->pic.pict_type = AV_PICTURE_TYPE_P;

    }



    if (code & 0x40) {

        uint32_t *out = (uint32_t *)c->pic.data[1];

        int index, count;



        index = bytestream2_get_byte(&c->gb);

        count = bytestream2_get_byte(&c->gb) + 1;



        if (index + count > AVPALETTE_SIZE)

            return AVERROR_INVALIDDATA;

        if (bytestream2_get_bytes_left(&c->gb) < 3 * AVPALETTE_SIZE)

            return AVERROR_INVALIDDATA;



        out += index;

        for (i = 0; i < count; i++) {

            unsigned r, g, b;



            r = bytestream2_get_byteu(&c->gb);

            r = r << 2 | r >> 4;

            g = bytestream2_get_byteu(&c->gb);

            g = g << 2 | g >> 4;

            b = bytestream2_get_byteu(&c->gb);

            b = b << 2 | b >> 4;

            *out++ = 0xFFU << 24 | r << 16 | g << 8 | b;

        }

        c->pic.palette_has_changed = 1;

    }



    switch (code & 0x0F) {

    case 0:

        if ((ret = decode_0(avctx, code, pkt->data)) < 0)

            return ret;

        break;

    case 1:

        dst = c->frame[c->current_frame];

        bytestream2_skip(&c->gb, 2);

        if (bytestream2_get_bytes_left(&c->gb) < c->video_size)

            return AVERROR_INVALIDDATA;

        bytestream2_get_bufferu(&c->gb, dst, c->video_size);

        break;

    case 2:

        frame = bytestream2_get_byte(&c->gb);

        if (frame > 3)

            return AVERROR_INVALIDDATA;

        if (frame != c->current_frame)

            memcpy(c->frame[c->current_frame], c->frame[frame], c->frame_size);

        break;

    case 4:

        dst = c->frame[c->current_frame];

        end = dst + c->video_size;



        bytestream2_skip(&c->gb, 2);



        while (dst < end) {

            int8_t code;

            int count;



            if (bytestream2_get_bytes_left(&c->gb) < 2)

                return AVERROR_INVALIDDATA;



            code  = bytestream2_get_byteu(&c->gb);

            count = FFABS(code) + 1;



            if (dst + count > end)

                return AVERROR_INVALIDDATA;

            if (code < 0)

                memset(dst, bytestream2_get_byteu(&c->gb), count);

            else

                bytestream2_get_buffer(&c->gb, dst, count);

            dst += count;

        }

        break;

    default:

        av_log_ask_for_sample(avctx, "unknown/invalid code\n");

        return AVERROR_INVALIDDATA;

    }



    dst = c->pic.data[0];

    src = c->frame[c->current_frame];

    for (i = 0; i < avctx->height; i++) {

        memcpy(dst, src, avctx->width);

        dst += c->pic.linesize[0];

        src += avctx->width;

    }



    c->current_frame = (c->current_frame + 1) & 3;



    *data_size       = sizeof(AVFrame);

    *(AVFrame *)data = c->pic;



    return pkt->size;

}
