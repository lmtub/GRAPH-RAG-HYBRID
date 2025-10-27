static int decode_hextile(VmncContext *c, uint8_t* dst, uint8_t* src, int w, int h, int stride)

{

    int i, j, k;

    int bg = 0, fg = 0, rects, color, flags, xy, wh;

    const int bpp = c->bpp2;

    uint8_t *dst2;

    int bw = 16, bh = 16;

    uint8_t *ssrc=src;



    for(j = 0; j < h; j += 16) {

        dst2 = dst;

        bw = 16;

        if(j + 16 > h) bh = h - j;

        for(i = 0; i < w; i += 16, dst2 += 16 * bpp) {

            if(i + 16 > w) bw = w - i;

            flags = *src++;

            if(flags & HT_RAW) {

                paint_raw(dst2, bw, bh, src, bpp, c->bigendian, stride);

                src += bw * bh * bpp;

            } else {

                if(flags & HT_BKG) {

                    bg = vmnc_get_pixel(src, bpp, c->bigendian); src += bpp;

                }

                if(flags & HT_FG) {

                    fg = vmnc_get_pixel(src, bpp, c->bigendian); src += bpp;

                }

                rects = 0;

                if(flags & HT_SUB)

                    rects = *src++;

                color = (flags & HT_CLR);



                paint_rect(dst2, 0, 0, bw, bh, bg, bpp, stride);



                for(k = 0; k < rects; k++) {

                    if(color) {

                        fg = vmnc_get_pixel(src, bpp, c->bigendian); src += bpp;

                    }

                    xy = *src++;

                    wh = *src++;

                    paint_rect(dst2, xy >> 4, xy & 0xF, (wh>>4)+1, (wh & 0xF)+1, fg, bpp, stride);

                }

            }

        }

        dst += stride * 16;

    }

    return src - ssrc;

}
