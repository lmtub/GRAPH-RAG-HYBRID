static int msrle_decode_init(AVCodecContext *avctx)

{

    MsrleContext *s = (MsrleContext *)avctx->priv_data;

    int i, j;

    unsigned char *palette;



    s->avctx = avctx;



    avctx->pix_fmt = PIX_FMT_PAL8;

    avctx->has_b_frames = 0;

    s->frame.data[0] = s->prev_frame.data[0] = NULL;



    /* convert palette */

    palette = (unsigned char *)s->avctx->extradata;

    memset (s->palette, 0, 256 * 4);

    for (i = 0, j = 0; i < s->avctx->extradata_size / 4; i++, j += 4)

        s->palette[i] = 

            (palette[j + 2] << 16) |

            (palette[j + 1] <<  8) |

            (palette[j + 0] <<  0);



    return 0;

}
