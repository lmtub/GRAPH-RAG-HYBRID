static int tiff_unpack_strip(TiffContext *s, uint8_t *dst, int stride,

                             const uint8_t *src, int size, int lines)

{

    int c, line, pixels, code, ret;

    const uint8_t *ssrc = src;

    int width           = ((s->width * s->bpp) + 7) >> 3;



    if (size <= 0)

        return AVERROR_INVALIDDATA;



    if (s->compr == TIFF_DEFLATE || s->compr == TIFF_ADOBE_DEFLATE) {

#if CONFIG_ZLIB

        return tiff_unpack_zlib(s, dst, stride, src, size, width, lines);

#else

        av_log(s->avctx, AV_LOG_ERROR,

               "zlib support not enabled, "

               "deflate compression not supported\n");

        return AVERROR(ENOSYS);

#endif

    }

    if (s->compr == TIFF_LZW) {

        if ((ret = ff_lzw_decode_init(s->lzw, 8, src, size, FF_LZW_TIFF)) < 0) {

            av_log(s->avctx, AV_LOG_ERROR, "Error initializing LZW decoder\n");

            return ret;

        }

    }

    if (s->compr == TIFF_CCITT_RLE ||

        s->compr == TIFF_G3        ||

        s->compr == TIFF_G4) {

        return tiff_unpack_fax(s, dst, stride, src, size, lines);

    }

    for (line = 0; line < lines; line++) {

        if (src - ssrc > size) {

            av_log(s->avctx, AV_LOG_ERROR, "Source data overread\n");

            return AVERROR_INVALIDDATA;

        }

        switch (s->compr) {

        case TIFF_RAW:

            if (ssrc + size - src < width)

                return AVERROR_INVALIDDATA;

            if (!s->fill_order) {

                memcpy(dst, src, width);

            } else {

                int i;

                for (i = 0; i < width; i++)

                    dst[i] = ff_reverse[src[i]];

            }

            src += width;

            break;

        case TIFF_PACKBITS:

            for (pixels = 0; pixels < width;) {

                code = (int8_t) *src++;

                if (code >= 0) {

                    code++;

                    if (pixels + code > width) {

                        av_log(s->avctx, AV_LOG_ERROR,

                               "Copy went out of bounds\n");

                        return AVERROR_INVALIDDATA;

                    }

                    memcpy(dst + pixels, src, code);

                    src    += code;

                    pixels += code;

                } else if (code != -128) { // -127..-1

                    code = (-code) + 1;

                    if (pixels + code > width) {

                        av_log(s->avctx, AV_LOG_ERROR,

                               "Run went out of bounds\n");

                        return AVERROR_INVALIDDATA;

                    }

                    c = *src++;

                    memset(dst + pixels, c, code);

                    pixels += code;

                }

            }

            break;

        case TIFF_LZW:

            pixels = ff_lzw_decode(s->lzw, dst, width);

            if (pixels < width) {

                av_log(s->avctx, AV_LOG_ERROR, "Decoded only %i bytes of %i\n",

                       pixels, width);

                return AVERROR_INVALIDDATA;

            }

            break;

        }

        dst += stride;

    }

    return 0;

}
