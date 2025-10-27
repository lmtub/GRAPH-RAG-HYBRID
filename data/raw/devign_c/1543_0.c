static int decode_frame(AVCodecContext *avctx,

                        void *data, int *got_frame,

                        AVPacket *avpkt)

{

    PNGDecContext *const s = avctx->priv_data;

    const uint8_t *buf     = avpkt->data;

    int buf_size           = avpkt->size;

    AVFrame *p             = data;

    uint8_t *crow_buf_base = NULL;

    uint32_t tag, length;

    int ret;



    /* check signature */

    if (buf_size < 8 ||

        memcmp(buf, ff_pngsig, 8) != 0 &&

        memcmp(buf, ff_mngsig, 8) != 0)

        return -1;



    bytestream2_init(&s->gb, buf + 8, buf_size - 8);

    s->y = s->state = 0;



    /* init the zlib */

    s->zstream.zalloc = ff_png_zalloc;

    s->zstream.zfree  = ff_png_zfree;

    s->zstream.opaque = NULL;

    ret = inflateInit(&s->zstream);

    if (ret != Z_OK)

        return -1;

    for (;;) {

        if (bytestream2_get_bytes_left(&s->gb) <= 0)

            goto fail;

        length = bytestream2_get_be32(&s->gb);

        if (length > 0x7fffffff)

            goto fail;

        tag = bytestream2_get_le32(&s->gb);

        av_dlog(avctx, "png: tag=%c%c%c%c length=%u\n",

                (tag & 0xff),

                ((tag >> 8) & 0xff),

                ((tag >> 16) & 0xff),

                ((tag >> 24) & 0xff), length);

        switch (tag) {

        case MKTAG('I', 'H', 'D', 'R'):

            if (length != 13)

                goto fail;

            s->width  = bytestream2_get_be32(&s->gb);

            s->height = bytestream2_get_be32(&s->gb);

            if (av_image_check_size(s->width, s->height, 0, avctx)) {

                s->width = s->height = 0;

                goto fail;

            }

            s->bit_depth        = bytestream2_get_byte(&s->gb);

            s->color_type       = bytestream2_get_byte(&s->gb);

            s->compression_type = bytestream2_get_byte(&s->gb);

            s->filter_type      = bytestream2_get_byte(&s->gb);

            s->interlace_type   = bytestream2_get_byte(&s->gb);

            bytestream2_skip(&s->gb, 4); /* crc */

            s->state |= PNG_IHDR;

            av_dlog(avctx, "width=%d height=%d depth=%d color_type=%d "

                           "compression_type=%d filter_type=%d interlace_type=%d\n",

                    s->width, s->height, s->bit_depth, s->color_type,

                    s->compression_type, s->filter_type, s->interlace_type);

            break;

        case MKTAG('I', 'D', 'A', 'T'):

            if (!(s->state & PNG_IHDR))

                goto fail;

            if (!(s->state & PNG_IDAT)) {

                /* init image info */

                avctx->width  = s->width;

                avctx->height = s->height;



                s->channels       = ff_png_get_nb_channels(s->color_type);

                s->bits_per_pixel = s->bit_depth * s->channels;

                s->bpp            = (s->bits_per_pixel + 7) >> 3;

                s->row_size       = (avctx->width * s->bits_per_pixel + 7) >> 3;



                if (s->bit_depth == 8 &&

                    s->color_type == PNG_COLOR_TYPE_RGB) {

                    avctx->pix_fmt = AV_PIX_FMT_RGB24;

                } else if (s->bit_depth == 8 &&

                           s->color_type == PNG_COLOR_TYPE_RGB_ALPHA) {

                    avctx->pix_fmt = AV_PIX_FMT_RGB32;

                } else if (s->bit_depth == 8 &&

                           s->color_type == PNG_COLOR_TYPE_GRAY) {

                    avctx->pix_fmt = AV_PIX_FMT_GRAY8;

                } else if (s->bit_depth == 16 &&

                           s->color_type == PNG_COLOR_TYPE_GRAY) {

                    avctx->pix_fmt = AV_PIX_FMT_GRAY16BE;

                } else if (s->bit_depth == 16 &&

                           s->color_type == PNG_COLOR_TYPE_RGB) {

                    avctx->pix_fmt = AV_PIX_FMT_RGB48BE;

                } else if (s->bit_depth == 1 &&

                           s->color_type == PNG_COLOR_TYPE_GRAY) {

                    avctx->pix_fmt = AV_PIX_FMT_MONOBLACK;

                } else if (s->bit_depth == 8 &&

                           s->color_type == PNG_COLOR_TYPE_PALETTE) {

                    avctx->pix_fmt = AV_PIX_FMT_PAL8;

                } else if (s->bit_depth == 8 &&

                           s->color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {

                    avctx->pix_fmt = AV_PIX_FMT_YA8;

                } else if (s->bit_depth == 16 &&

                           s->color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {

                    avctx->pix_fmt = AV_PIX_FMT_YA16BE;

                } else {

                    goto fail;

                }



                if (ff_get_buffer(avctx, p, AV_GET_BUFFER_FLAG_REF) < 0) {

                    av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

                    goto fail;

                }

                p->pict_type        = AV_PICTURE_TYPE_I;

                p->key_frame        = 1;

                p->interlaced_frame = !!s->interlace_type;



                /* compute the compressed row size */

                if (!s->interlace_type) {

                    s->crow_size = s->row_size + 1;

                } else {

                    s->pass          = 0;

                    s->pass_row_size = ff_png_pass_row_size(s->pass,

                                                            s->bits_per_pixel,

                                                            s->width);

                    s->crow_size = s->pass_row_size + 1;

                }

                av_dlog(avctx, "row_size=%d crow_size =%d\n",

                        s->row_size, s->crow_size);

                s->image_buf      = p->data[0];

                s->image_linesize = p->linesize[0];

                /* copy the palette if needed */

                if (s->color_type == PNG_COLOR_TYPE_PALETTE)

                    memcpy(p->data[1], s->palette, 256 * sizeof(uint32_t));

                /* empty row is used if differencing to the first row */

                s->last_row = av_mallocz(s->row_size);

                if (!s->last_row)

                    goto fail;

                if (s->interlace_type ||

                    s->color_type == PNG_COLOR_TYPE_RGB_ALPHA) {

                    s->tmp_row = av_malloc(s->row_size);

                    if (!s->tmp_row)

                        goto fail;

                }

                /* compressed row */

                crow_buf_base = av_malloc(s->row_size + 16);

                if (!crow_buf_base)

                    goto fail;



                /* we want crow_buf+1 to be 16-byte aligned */

                s->crow_buf          = crow_buf_base + 15;

                s->zstream.avail_out = s->crow_size;

                s->zstream.next_out  = s->crow_buf;

            }

            s->state |= PNG_IDAT;

            if (png_decode_idat(s, length) < 0)

                goto fail;

            bytestream2_skip(&s->gb, 4); /* crc */

            break;

        case MKTAG('P', 'L', 'T', 'E'):

        {

            int n, i, r, g, b;



            if ((length % 3) != 0 || length > 256 * 3)

                goto skip_tag;

            /* read the palette */

            n = length / 3;

            for (i = 0; i < n; i++) {

                r = bytestream2_get_byte(&s->gb);

                g = bytestream2_get_byte(&s->gb);

                b = bytestream2_get_byte(&s->gb);

                s->palette[i] = (0xff << 24) | (r << 16) | (g << 8) | b;

            }

            for (; i < 256; i++)

                s->palette[i] = (0xff << 24);

            s->state |= PNG_PLTE;

            bytestream2_skip(&s->gb, 4);     /* crc */

        }

        break;

        case MKTAG('t', 'R', 'N', 'S'):

        {

            int v, i;



            /* read the transparency. XXX: Only palette mode supported */

            if (s->color_type != PNG_COLOR_TYPE_PALETTE ||

                length > 256 ||

                !(s->state & PNG_PLTE))

                goto skip_tag;

            for (i = 0; i < length; i++) {

                v = bytestream2_get_byte(&s->gb);

                s->palette[i] = (s->palette[i] & 0x00ffffff) | (v << 24);

            }

            bytestream2_skip(&s->gb, 4);     /* crc */

        }

        break;

        case MKTAG('I', 'E', 'N', 'D'):

            if (!(s->state & PNG_ALLIMAGE))

                goto fail;

            bytestream2_skip(&s->gb, 4); /* crc */

            goto exit_loop;

        default:

            /* skip tag */

skip_tag:

            bytestream2_skip(&s->gb, length + 4);

            break;

        }

    }

exit_loop:

    /* handle p-frames only if a predecessor frame is available */

    if (s->prev->data[0]) {

        if (!(avpkt->flags & AV_PKT_FLAG_KEY)) {

            int i, j;

            uint8_t *pd      = p->data[0];

            uint8_t *pd_last = s->prev->data[0];



            for (j = 0; j < s->height; j++) {

                for (i = 0; i < s->width * s->bpp; i++)

                    pd[i] += pd_last[i];

                pd      += s->image_linesize;

                pd_last += s->image_linesize;

            }

        }

    }



    av_frame_unref(s->prev);

    if ((ret = av_frame_ref(s->prev, p)) < 0)

        goto fail;



    *got_frame = 1;



    ret = bytestream2_tell(&s->gb);

the_end:

    inflateEnd(&s->zstream);

    av_free(crow_buf_base);

    s->crow_buf = NULL;

    av_freep(&s->last_row);

    av_freep(&s->tmp_row);

    return ret;

fail:

    ret = -1;

    goto the_end;

}
