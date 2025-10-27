static int decode_frame(AVCodecContext * avctx, void *data, int *data_size, AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    KmvcContext *const ctx = avctx->priv_data;

    uint8_t *out, *src;

    int i;

    int header;

    int blocksize;

    const uint8_t *pal = av_packet_get_side_data(avpkt, AV_PKT_DATA_PALETTE, NULL);



    if (ctx->pic.data[0])

        avctx->release_buffer(avctx, &ctx->pic);



    ctx->pic.reference = 1;

    ctx->pic.buffer_hints = FF_BUFFER_HINTS_VALID;

    if (avctx->get_buffer(avctx, &ctx->pic) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return -1;

    }



    header = *buf++;



    /* blocksize 127 is really palette change event */

    if (buf[0] == 127) {

        buf += 3;

        for (i = 0; i < 127; i++) {

            ctx->pal[i + (header & 0x81)] = AV_RB24(buf);

            buf += 4;

        }

        buf -= 127 * 4 + 3;

    }



    if (header & KMVC_KEYFRAME) {

        ctx->pic.key_frame = 1;

        ctx->pic.pict_type = AV_PICTURE_TYPE_I;

    } else {

        ctx->pic.key_frame = 0;

        ctx->pic.pict_type = AV_PICTURE_TYPE_P;

    }



    if (header & KMVC_PALETTE) {

        ctx->pic.palette_has_changed = 1;

        // palette starts from index 1 and has 127 entries

        for (i = 1; i <= ctx->palsize; i++) {

            ctx->pal[i] = bytestream_get_be24(&buf);

        }

    }



    if (pal) {

        ctx->pic.palette_has_changed = 1;

        memcpy(ctx->pal, pal, AVPALETTE_SIZE);

    }



    if (ctx->setpal) {

        ctx->setpal = 0;

        ctx->pic.palette_has_changed = 1;

    }



    /* make the palette available on the way out */

    memcpy(ctx->pic.data[1], ctx->pal, 1024);



    blocksize = *buf++;



    if (blocksize != 8 && blocksize != 127) {

        av_log(avctx, AV_LOG_ERROR, "Block size = %i\n", blocksize);

        return -1;

    }

    memset(ctx->cur, 0, 320 * 200);

    switch (header & KMVC_METHOD) {

    case 0:

    case 1: // used in palette changed event

        memcpy(ctx->cur, ctx->prev, 320 * 200);

        break;

    case 3:

        kmvc_decode_intra_8x8(ctx, buf, buf_size, avctx->width, avctx->height);

        break;

    case 4:

        kmvc_decode_inter_8x8(ctx, buf, buf_size, avctx->width, avctx->height);

        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "Unknown compression method %i\n", header & KMVC_METHOD);

        return -1;

    }



    out = ctx->pic.data[0];

    src = ctx->cur;

    for (i = 0; i < avctx->height; i++) {

        memcpy(out, src, avctx->width);

        src += 320;

        out += ctx->pic.linesize[0];

    }



    /* flip buffers */

    if (ctx->cur == ctx->frm0) {

        ctx->cur = ctx->frm1;

        ctx->prev = ctx->frm0;

    } else {

        ctx->cur = ctx->frm0;

        ctx->prev = ctx->frm1;

    }



    *data_size = sizeof(AVFrame);

    *(AVFrame *) data = ctx->pic;



    /* always report that the buffer was completely consumed */

    return buf_size;

}
