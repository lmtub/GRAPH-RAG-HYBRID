static int gif_encode_frame(AVCodecContext *avctx, unsigned char *outbuf, int buf_size, void *data)

{

    GIFContext *s = avctx->priv_data;

    AVFrame *pict = data;

    AVFrame *const p = (AVFrame *)&s->picture;

    uint8_t *outbuf_ptr = outbuf;



    *p = *pict;

    p->pict_type = FF_I_TYPE;

    p->key_frame = 1;

    gif_image_write_header(&outbuf_ptr, avctx->width, avctx->height, -1, (uint32_t *)pict->data[1]);

    gif_image_write_image(&outbuf_ptr, 0, 0, avctx->width, avctx->height, pict->data[0], pict->linesize[0], PIX_FMT_PAL8);

    return outbuf_ptr - outbuf;

}
