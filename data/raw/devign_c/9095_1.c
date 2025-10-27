static int gif_image_write_image(AVCodecContext *avctx,

                                 uint8_t **bytestream, uint8_t *end,

                                 const uint8_t *buf, int linesize)

{

    GIFContext *s = avctx->priv_data;

    int len, height;

    const uint8_t *ptr;

    /* image block */



    bytestream_put_byte(bytestream, 0x2c);

    bytestream_put_le16(bytestream, 0);

    bytestream_put_le16(bytestream, 0);

    bytestream_put_le16(bytestream, avctx->width);

    bytestream_put_le16(bytestream, avctx->height);

    bytestream_put_byte(bytestream, 0x00); /* flags */

    /* no local clut */



    bytestream_put_byte(bytestream, 0x08);



    ff_lzw_encode_init(s->lzw, s->buf, avctx->width*avctx->height,

                       12, FF_LZW_GIF, put_bits);



    ptr = buf;

    for (height = avctx->height; height--;) {

        len += ff_lzw_encode(s->lzw, ptr, avctx->width);

        ptr += linesize;

    }

    len += ff_lzw_encode_flush(s->lzw, flush_put_bits);



    ptr = s->buf;

    while (len > 0) {

        int size = FFMIN(255, len);

        bytestream_put_byte(bytestream, size);

        if (end - *bytestream < size)

            return -1;

        bytestream_put_buffer(bytestream, ptr, size);

        ptr += size;

        len -= size;

    }

    bytestream_put_byte(bytestream, 0x00); /* end of image block */

    bytestream_put_byte(bytestream, 0x3b);

    return 0;

}
