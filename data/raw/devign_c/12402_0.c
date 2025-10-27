static int targa_encode_frame(AVCodecContext *avctx, AVPacket *pkt,

                              const AVFrame *p, int *got_packet)

{

    int bpp, picsize, datasize = -1, ret;

    uint8_t *out;



    if(avctx->width > 0xffff || avctx->height > 0xffff) {

        av_log(avctx, AV_LOG_ERROR, "image dimensions too large\n");

        return AVERROR(EINVAL);

    }

    picsize = av_image_get_buffer_size(avctx->pix_fmt,

                                       avctx->width, avctx->height, 1);

    if ((ret = ff_alloc_packet(pkt, picsize + 45)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "encoded frame too large\n");

        return ret;

    }



    /* zero out the header and only set applicable fields */

    memset(pkt->data, 0, 12);

    AV_WL16(pkt->data+12, avctx->width);

    AV_WL16(pkt->data+14, avctx->height);

    /* image descriptor byte: origin is always top-left, bits 0-3 specify alpha */

    pkt->data[17] = 0x20 | (avctx->pix_fmt == AV_PIX_FMT_BGRA ? 8 : 0);



    switch(avctx->pix_fmt) {

    case AV_PIX_FMT_GRAY8:

        pkt->data[2]  = TGA_BW;     /* uncompressed grayscale image */

        pkt->data[16] = 8;          /* bpp */

        break;

    case AV_PIX_FMT_RGB555LE:

        pkt->data[2]  = TGA_RGB;    /* uncompresses true-color image */

        pkt->data[16] = 16;         /* bpp */

        break;

    case AV_PIX_FMT_BGR24:

        pkt->data[2]  = TGA_RGB;    /* uncompressed true-color image */

        pkt->data[16] = 24;         /* bpp */

        break;

    case AV_PIX_FMT_BGRA:

        pkt->data[2]  = TGA_RGB;    /* uncompressed true-color image */

        pkt->data[16] = 32;         /* bpp */

        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "Pixel format '%s' not supported.\n",

               av_get_pix_fmt_name(avctx->pix_fmt));

        return AVERROR(EINVAL);

    }

    bpp = pkt->data[16] >> 3;



    out = pkt->data + 18;  /* skip past the header we just output */



    /* try RLE compression */

    if (avctx->coder_type != FF_CODER_TYPE_RAW)

        datasize = targa_encode_rle(out, picsize, p, bpp, avctx->width, avctx->height);



    /* if that worked well, mark the picture as RLE compressed */

    if(datasize >= 0)

        pkt->data[2] |= 8;



    /* if RLE didn't make it smaller, go back to no compression */

    else datasize = targa_encode_normal(out, p, bpp, avctx->width, avctx->height);



    out += datasize;



    /* The standard recommends including this section, even if we don't use

     * any of the features it affords. TODO: take advantage of the pixel

     * aspect ratio and encoder ID fields available? */

    memcpy(out, "\0\0\0\0\0\0\0\0TRUEVISION-XFILE.", 26);



    pkt->size   = out + 26 - pkt->data;

    pkt->flags |= AV_PKT_FLAG_KEY;

    *got_packet = 1;



    return 0;

}
