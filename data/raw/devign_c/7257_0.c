static int targa_encode_frame(AVCodecContext *avctx,

                              unsigned char *outbuf,

                              int buf_size, void *data){

    AVFrame *p = data;

    int bpp, picsize, datasize;

    uint8_t *out;



    if(avctx->width > 0xffff || avctx->height > 0xffff) {

        av_log(avctx, AV_LOG_ERROR, "image dimensions too large\n");

        return -1;

    }

    picsize = avpicture_get_size(avctx->pix_fmt, avctx->width, avctx->height);

    if(buf_size < picsize + 45) {

        av_log(avctx, AV_LOG_ERROR, "encoded frame too large\n");

        return -1;

    }



    p->pict_type= FF_I_TYPE;

    p->key_frame= 1;



    /* zero out the header and only set applicable fields */

    memset(outbuf, 0, 11);

    AV_WL16(outbuf+12, avctx->width);

    AV_WL16(outbuf+14, avctx->height);

    outbuf[17] = 0x20;           /* origin is top-left. no alpha */



    /* TODO: support alpha channel */

    switch(avctx->pix_fmt) {

    case PIX_FMT_GRAY8:

        outbuf[2] = 3;           /* uncompressed grayscale image */

        outbuf[16] = 8;          /* bpp */

        break;

    case PIX_FMT_RGB555:

        outbuf[2] = 2;           /* uncompresses true-color image */

        outbuf[16] = 16;         /* bpp */

        break;

    case PIX_FMT_BGR24:

        outbuf[2] = 2;           /* uncompressed true-color image */

        outbuf[16] = 24;         /* bpp */

        break;

    default:

        return -1;

    }

    bpp = outbuf[16] >> 3;



    out = outbuf + 18;  /* skip past the header we just output */



    /* try RLE compression */

    datasize = targa_encode_rle(out, picsize, p, bpp, avctx->width, avctx->height);



    /* if that worked well, mark the picture as RLE compressed */

    if(datasize >= 0)

        outbuf[2] |= 8;



    /* if RLE didn't make it smaller, go back to no compression */

    else datasize = targa_encode_normal(out, p, bpp, avctx->width, avctx->height);



    out += datasize;



    /* The standard recommends including this section, even if we don't use

     * any of the features it affords. TODO: take advantage of the pixel

     * aspect ratio and encoder ID fields available? */

    memcpy(out, "\0\0\0\0\0\0\0\0TRUEVISION-XFILE.", 26);



    return out + 26 - outbuf;

}
