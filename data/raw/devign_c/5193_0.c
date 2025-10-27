static void inline xan_wc3_build_palette(XanContext *s, 

    unsigned int *palette_data)

{

    int i;

    unsigned char r, g, b;

    unsigned short *palette16;

    unsigned int *palette32;

    unsigned int pal_elem;



    /* transform the palette passed through the palette control structure

     * into the necessary internal format depending on colorspace */



    switch (s->avctx->pix_fmt) {



    case PIX_FMT_RGB555:

        palette16 = (unsigned short *)s->palette;

        for (i = 0; i < PALETTE_COUNT; i++) {

            pal_elem = palette_data[i];

            r = (pal_elem >> 16) & 0xff;

            g = (pal_elem >> 8) & 0xff;

            b = pal_elem & 0xff;

            palette16[i] = 

                ((r >> 3) << 10) |

                ((g >> 3) <<  5) |

                ((b >> 3) <<  0);

        }

        break;



    case PIX_FMT_RGB565:

        palette16 = (unsigned short *)s->palette;

        for (i = 0; i < PALETTE_COUNT; i++) {

            pal_elem = palette_data[i];

            r = (pal_elem >> 16) & 0xff;

            g = (pal_elem >> 8) & 0xff;

            b = pal_elem & 0xff;

            palette16[i] = 

                ((r >> 3) << 11) |

                ((g >> 2) <<  5) |

                ((b >> 3) <<  0);

        }

        break;



    case PIX_FMT_RGB24:

        for (i = 0; i < PALETTE_COUNT; i++) {

            pal_elem = palette_data[i];

            r = (pal_elem >> 16) & 0xff;

            g = (pal_elem >> 8) & 0xff;

            b = pal_elem & 0xff;

            s->palette[i * 4 + 0] = r;

            s->palette[i * 4 + 1] = g;

            s->palette[i * 4 + 2] = b;

        }

        break;



    case PIX_FMT_BGR24:

        for (i = 0; i < PALETTE_COUNT; i++) {

            pal_elem = palette_data[i];

            r = (pal_elem >> 16) & 0xff;

            g = (pal_elem >> 8) & 0xff;

            b = pal_elem & 0xff;

            s->palette[i * 4 + 0] = b;

            s->palette[i * 4 + 1] = g;

            s->palette[i * 4 + 2] = r;

        }

        break;



    case PIX_FMT_PAL8:

    case PIX_FMT_RGBA32:

        palette32 = (unsigned int *)s->palette;

        memcpy (palette32, palette_data, PALETTE_COUNT * sizeof(unsigned int));

        break;



    case PIX_FMT_YUV444P:

        for (i = 0; i < PALETTE_COUNT; i++) {

            pal_elem = palette_data[i];

            r = (pal_elem >> 16) & 0xff;

            g = (pal_elem >> 8) & 0xff;

            b = pal_elem & 0xff;

            s->palette[i * 4 + 0] = COMPUTE_Y(r, g, b);

            s->palette[i * 4 + 1] = COMPUTE_U(r, g, b);

            s->palette[i * 4 + 2] = COMPUTE_V(r, g, b);

        }

        break;



    default:

        av_log(s->avctx, AV_LOG_ERROR, " Xan WC3: Unhandled colorspace\n");

        break;

    }

}
