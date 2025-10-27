static int decode_fctl_chunk(AVCodecContext *avctx, PNGDecContext *s,

                             uint32_t length)

{

    uint32_t sequence_number;



    if (length != 26)

        return AVERROR_INVALIDDATA;



    if (!(s->state & PNG_IHDR)) {

        av_log(avctx, AV_LOG_ERROR, "fctl before IHDR\n");

        return AVERROR_INVALIDDATA;

    }



    s->last_w = s->cur_w;

    s->last_h = s->cur_h;

    s->last_x_offset = s->x_offset;

    s->last_y_offset = s->y_offset;

    s->last_dispose_op = s->dispose_op;



    sequence_number = bytestream2_get_be32(&s->gb);

    s->cur_w        = bytestream2_get_be32(&s->gb);

    s->cur_h        = bytestream2_get_be32(&s->gb);

    s->x_offset     = bytestream2_get_be32(&s->gb);

    s->y_offset     = bytestream2_get_be32(&s->gb);

    bytestream2_skip(&s->gb, 4); /* delay_num (2), delay_den (2) */

    s->dispose_op   = bytestream2_get_byte(&s->gb);

    s->blend_op     = bytestream2_get_byte(&s->gb);

    bytestream2_skip(&s->gb, 4); /* crc */



    if (sequence_number == 0 &&

        (s->cur_w != s->width ||

         s->cur_h != s->height ||

         s->x_offset != 0 ||

         s->y_offset != 0) ||

        s->cur_w <= 0 || s->cur_h <= 0 ||

        s->x_offset < 0 || s->y_offset < 0 ||

        s->cur_w > s->width - s->x_offset|| s->cur_h > s->height - s->y_offset)

            return AVERROR_INVALIDDATA;



    if (sequence_number == 0 && s->dispose_op == APNG_DISPOSE_OP_PREVIOUS) {

        // No previous frame to revert to for the first frame

        // Spec says to just treat it as a APNG_DISPOSE_OP_BACKGROUND

        s->dispose_op = APNG_DISPOSE_OP_BACKGROUND;

    }



    if (s->dispose_op == APNG_BLEND_OP_OVER && !s->has_trns && (

            avctx->pix_fmt == AV_PIX_FMT_RGB24 ||

            avctx->pix_fmt == AV_PIX_FMT_RGB48BE ||

            avctx->pix_fmt == AV_PIX_FMT_PAL8 ||

            avctx->pix_fmt == AV_PIX_FMT_GRAY8 ||

            avctx->pix_fmt == AV_PIX_FMT_GRAY16BE ||

            avctx->pix_fmt == AV_PIX_FMT_MONOBLACK

        )) {

        // APNG_DISPOSE_OP_OVER is the same as APNG_DISPOSE_OP_SOURCE when there is no alpha channel

        s->dispose_op = APNG_BLEND_OP_SOURCE;

    }



    return 0;

}
