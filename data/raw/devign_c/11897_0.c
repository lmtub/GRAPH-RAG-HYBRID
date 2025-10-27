static int pcx_decode_frame(AVCodecContext *avctx, void *data, int *got_frame,

                            AVPacket *avpkt)

{

    PCXContext * const s = avctx->priv_data;

    AVFrame *picture = data;

    AVFrame * const p = &s->picture;

    GetByteContext gb;

    int compressed, xmin, ymin, xmax, ymax, ret;

    unsigned int w, h, bits_per_pixel, bytes_per_line, nplanes, stride, y, x,

                 bytes_per_scanline;

    uint8_t *ptr, *scanline;



    if (avpkt->size < 128)

        return AVERROR_INVALIDDATA;



    bytestream2_init(&gb, avpkt->data, avpkt->size);



    if (bytestream2_get_byteu(&gb) != 0x0a || bytestream2_get_byteu(&gb) > 5) {

        av_log(avctx, AV_LOG_ERROR, "this is not PCX encoded data\n");

        return AVERROR_INVALIDDATA;

    }



    compressed = bytestream2_get_byteu(&gb);

    bits_per_pixel = bytestream2_get_byteu(&gb);

    xmin = bytestream2_get_le16u(&gb);

    ymin = bytestream2_get_le16u(&gb);

    xmax = bytestream2_get_le16u(&gb);

    ymax = bytestream2_get_le16u(&gb);

    avctx->sample_aspect_ratio.num = bytestream2_get_le16u(&gb);

    avctx->sample_aspect_ratio.den = bytestream2_get_le16u(&gb);



    if (xmax < xmin || ymax < ymin) {

        av_log(avctx, AV_LOG_ERROR, "invalid image dimensions\n");

        return AVERROR_INVALIDDATA;

    }



    w = xmax - xmin + 1;

    h = ymax - ymin + 1;



    bytestream2_skipu(&gb, 49);

    nplanes            = bytestream2_get_byteu(&gb);

    bytes_per_line     = bytestream2_get_le16u(&gb);

    bytes_per_scanline = nplanes * bytes_per_line;



    if (bytes_per_scanline < (w * bits_per_pixel * nplanes + 7) / 8) {

        av_log(avctx, AV_LOG_ERROR, "PCX data is corrupted\n");

        return AVERROR_INVALIDDATA;

    }



    switch ((nplanes<<8) + bits_per_pixel) {

        case 0x0308:

            avctx->pix_fmt = AV_PIX_FMT_RGB24;

            break;

        case 0x0108:

        case 0x0104:

        case 0x0102:

        case 0x0101:

        case 0x0401:

        case 0x0301:

        case 0x0201:

            avctx->pix_fmt = AV_PIX_FMT_PAL8;

            break;

        default:

            av_log(avctx, AV_LOG_ERROR, "invalid PCX file\n");

            return AVERROR_INVALIDDATA;

    }



    bytestream2_skipu(&gb, 60);



    if (p->data[0])

        avctx->release_buffer(avctx, p);



    if ((ret = av_image_check_size(w, h, 0, avctx)) < 0)

        return ret;

    if (w != avctx->width || h != avctx->height)

        avcodec_set_dimensions(avctx, w, h);

    if ((ret = ff_get_buffer(avctx, p)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return ret;

    }



    p->pict_type = AV_PICTURE_TYPE_I;



    ptr    = p->data[0];

    stride = p->linesize[0];



    scanline = av_malloc(bytes_per_scanline);

    if (!scanline)

        return AVERROR(ENOMEM);



    if (nplanes == 3 && bits_per_pixel == 8) {

        for (y=0; y<h; y++) {

            pcx_rle_decode(&gb, scanline, bytes_per_scanline, compressed);



            for (x=0; x<w; x++) {

                ptr[3*x  ] = scanline[x                    ];

                ptr[3*x+1] = scanline[x+ bytes_per_line    ];

                ptr[3*x+2] = scanline[x+(bytes_per_line<<1)];

            }



            ptr += stride;

        }



    } else if (nplanes == 1 && bits_per_pixel == 8) {

        int palstart = avpkt->size - 769;



        for (y=0; y<h; y++, ptr+=stride) {

            pcx_rle_decode(&gb, scanline, bytes_per_scanline, compressed);

            memcpy(ptr, scanline, w);

        }



        if (bytestream2_tell(&gb) != palstart) {

            av_log(avctx, AV_LOG_WARNING, "image data possibly corrupted\n");

            bytestream2_seek(&gb, palstart, SEEK_SET);

        }

        if (bytestream2_get_byte(&gb) != 12) {

            av_log(avctx, AV_LOG_ERROR, "expected palette after image data\n");

            ret = AVERROR_INVALIDDATA;

            goto end;

        }



    } else if (nplanes == 1) {   /* all packed formats, max. 16 colors */

        GetBitContext s;



        for (y=0; y<h; y++) {

            init_get_bits8(&s, scanline, bytes_per_scanline);



            pcx_rle_decode(&gb, scanline, bytes_per_scanline, compressed);



            for (x=0; x<w; x++)

                ptr[x] = get_bits(&s, bits_per_pixel);

            ptr += stride;

        }



    } else {    /* planar, 4, 8 or 16 colors */

        int i;



        for (y=0; y<h; y++) {

            pcx_rle_decode(&gb, scanline, bytes_per_scanline, compressed);



            for (x=0; x<w; x++) {

                int m = 0x80 >> (x&7), v = 0;

                for (i=nplanes - 1; i>=0; i--) {

                    v <<= 1;

                    v  += !!(scanline[i*bytes_per_line + (x>>3)] & m);

                }

                ptr[x] = v;

            }

            ptr += stride;

        }

    }



    ret = bytestream2_tell(&gb);

    if (nplanes == 1 && bits_per_pixel == 8) {

        pcx_palette(&gb, (uint32_t *) p->data[1], 256);

        ret += 256 * 3;

    } else if (bits_per_pixel * nplanes == 1) {

        AV_WN32A(p->data[1]  , 0xFF000000);

        AV_WN32A(p->data[1]+4, 0xFFFFFFFF);

    } else if (bits_per_pixel < 8) {

        bytestream2_seek(&gb, 16, SEEK_SET);

        pcx_palette(&gb, (uint32_t *) p->data[1], 16);

    }



    *picture = s->picture;

    *got_frame = 1;



end:

    av_free(scanline);

    return ret;

}
