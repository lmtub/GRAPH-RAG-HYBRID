static int qtrle_decode_frame(AVCodecContext *avctx,

                              void *data, int *data_size,

                              AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    QtrleContext *s = avctx->priv_data;

    int header, start_line;

    int stream_ptr, height, row_ptr;

    int has_palette = 0;



    s->buf = buf;

    s->size = buf_size;



    s->frame.reference = 1;

    s->frame.buffer_hints = FF_BUFFER_HINTS_VALID | FF_BUFFER_HINTS_PRESERVE |

                            FF_BUFFER_HINTS_REUSABLE | FF_BUFFER_HINTS_READABLE;

    if (avctx->reget_buffer(avctx, &s->frame)) {

        av_log (s->avctx, AV_LOG_ERROR, "reget_buffer() failed\n");

        return -1;

    }



    /* check if this frame is even supposed to change */

    if (s->size < 8)

        goto done;



    /* start after the chunk size */

    stream_ptr = 4;



    /* fetch the header */

    header = AV_RB16(&s->buf[stream_ptr]);

    stream_ptr += 2;



    /* if a header is present, fetch additional decoding parameters */

    if (header & 0x0008) {

        if(s->size < 14)

            goto done;

        start_line = AV_RB16(&s->buf[stream_ptr]);

        stream_ptr += 4;

        height = AV_RB16(&s->buf[stream_ptr]);

        stream_ptr += 4;

    } else {

        start_line = 0;

        height = s->avctx->height;

    }

    row_ptr = s->frame.linesize[0] * start_line;



    switch (avctx->bits_per_coded_sample) {

    case 1:

    case 33:

        qtrle_decode_1bpp(s, stream_ptr, row_ptr, height);

        break;



    case 2:

    case 34:

        qtrle_decode_2n4bpp(s, stream_ptr, row_ptr, height, 2);

        has_palette = 1;

        break;



    case 4:

    case 36:

        qtrle_decode_2n4bpp(s, stream_ptr, row_ptr, height, 4);

        has_palette = 1;

        break;



    case 8:

    case 40:

        qtrle_decode_8bpp(s, stream_ptr, row_ptr, height);

        has_palette = 1;

        break;



    case 16:

        qtrle_decode_16bpp(s, stream_ptr, row_ptr, height);

        break;



    case 24:

        qtrle_decode_24bpp(s, stream_ptr, row_ptr, height);

        break;



    case 32:

        qtrle_decode_32bpp(s, stream_ptr, row_ptr, height);

        break;



    default:

        av_log (s->avctx, AV_LOG_ERROR, "Unsupported colorspace: %d bits/sample?\n",

            avctx->bits_per_coded_sample);

        break;

    }



    if(has_palette) {

        const uint8_t *pal = av_packet_get_side_data(avpkt, AV_PKT_DATA_PALETTE, NULL);



        if (pal) {

            s->frame.palette_has_changed = 1;

            memcpy(s->pal, pal, AVPALETTE_SIZE);

        }



        /* make the palette available on the way out */

        memcpy(s->frame.data[1], s->pal, AVPALETTE_SIZE);

    }



done:

    *data_size = sizeof(AVFrame);

    *(AVFrame*)data = s->frame;



    /* always report that the buffer was completely consumed */

    return buf_size;

}
