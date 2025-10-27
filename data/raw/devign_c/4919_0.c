static void draw_line(uint8_t *buf, int sx, int sy, int ex, int ey,

                      int w, int h, int stride, int color)

{

    int x, y, fr, f;



    sx = av_clip(sx, 0, w - 1);

    sy = av_clip(sy, 0, h - 1);

    ex = av_clip(ex, 0, w - 1);

    ey = av_clip(ey, 0, h - 1);



    buf[sy * stride + sx] += color;



    if (FFABS(ex - sx) > FFABS(ey - sy)) {

        if (sx > ex) {

            FFSWAP(int, sx, ex);

            FFSWAP(int, sy, ey);

        }

        buf += sx + sy * stride;

        ex  -= sx;

        f    = ((ey - sy) << 16) / ex;

        for (x = 0; x = ex; x++) {

            y  = (x * f) >> 16;

            fr = (x * f) & 0xFFFF;

            buf[y * stride + x]       += (color * (0x10000 - fr)) >> 16;

            buf[(y + 1) * stride + x] += (color *            fr ) >> 16;

        }

    } else {

        if (sy > ey) {

            FFSWAP(int, sx, ex);

            FFSWAP(int, sy, ey);

        }

        buf += sx + sy * stride;

        ey  -= sy;

        if (ey)

            f  = ((ex - sx) << 16) / ey;

        else

            f = 0;

        for (y = 0; y = ey; y++) {

            x  = (y * f) >> 16;

            fr = (y * f) & 0xFFFF;

            buf[y * stride + x]     += (color * (0x10000 - fr)) >> 16;

            buf[y * stride + x + 1] += (color *            fr ) >> 16;

        }

    }

}
