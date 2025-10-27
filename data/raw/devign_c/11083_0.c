static int decode_frame(AVCodecContext * avctx, void *data, int *got_frame,

                        AVPacket *avpkt)

{

    KmvcContext *const ctx = avctx->priv_data;

    AVFrame *frame = data;

    uint8_t *out, *src;

    int i, ret;

    int header;

    int blocksize;

    const uint8_t *pal = av_packet_get_side_data(avpkt, AV_PKT_DATA_PALETTE, NULL);



    bytestream2_init(&ctx->g, avpkt->data, avpkt->size);



    if ((ret = ff_get_buffer(avctx, frame, 0)) < 0)

        return ret;



    header = bytestream2_get_byte(&ctx->g);



    /* blocksize 127 is really palette change event */

    if (bytestream2_peek_byte(&ctx->g) == 127) {

        bytestream2_skip(&ctx->g, 3);

        for (i = 0; i < 127; i++) {

            ctx->pal[i + (header & 0x81)] = 0xFFU << 24 | bytestream2_get_be24(&ctx->g);

            bytestream2_skip(&ctx->g, 1);

        }

        bytestream2_seek(&ctx->g, -127 * 4 - 3, SEEK_CUR);

    }



    if (header & KMVC_KEYFRAME) {

        frame->key_frame = 1;

        frame->pict_type = AV_PICTURE_TYPE_I;

    } else {

        frame->key_frame = 0;

        frame->pict_type = AV_PICTURE_TYPE_P;

    }



    if (header & KMVC_PALETTE) {

        frame->palette_has_changed = 1;

        // palette starts from index 1 and has 127 entries

        for (i = 1; i <= ctx->palsize; i++) {

            ctx->pal[i] = 0xFFU << 24 | bytestream2_get_be24(&ctx->g);

        }

    }



    if (pal) {

        frame->palette_has_changed = 1;

        memcpy(ctx->pal, pal, AVPALETTE_SIZE);

    }



    if (ctx->setpal) {

        ctx->setpal = 0;

        frame->palette_has_changed = 1;

    }



    /* make the palette available on the way out */

    memcpy(frame->data[1], ctx->pal, 1024);



    blocksize = bytestream2_get_byte(&ctx->g);



    if (blocksize != 8 && blocksize != 127) {

        av_log(avctx, AV_LOG_ERROR, "Block size = %i\n", blocksize);

        return AVERROR_INVALIDDATA;

    }

    memset(ctx->cur, 0, 320 * 200);

    switch (header & KMVC_METHOD) {

    case 0:

    case 1: // used in palette changed event

        memcpy(ctx->cur, ctx->prev, 320 * 200);

        break;

    case 3:

        kmvc_decode_intra_8x8(ctx, avctx->width, avctx->height);

        break;

    case 4:

        kmvc_decode_inter_8x8(ctx, avctx->width, avctx->height);

        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "Unknown compression method %i\n", header & KMVC_METHOD);

        return AVERROR_INVALIDDATA;

    }



    out = frame->data[0];

    src = ctx->cur;

    for (i = 0; i < avctx->height; i++) {

        memcpy(out, src, avctx->width);

        src += 320;

        out += frame->linesize[0];

    }



    /* flip buffers */

    if (ctx->cur == ctx->frm0) {

        ctx->cur = ctx->frm1;

        ctx->prev = ctx->frm0;

    } else {

        ctx->cur = ctx->frm0;

        ctx->prev = ctx->frm1;

    }



    *got_frame = 1;



    /* always report that the buffer was completely consumed */

    return avpkt->size;

}
