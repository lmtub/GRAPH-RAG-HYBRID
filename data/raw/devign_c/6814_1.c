static void mp_decode_line(MotionPixelsContext *mp, GetBitContext *gb, int y)

{

    YuvPixel p;

    const int y0 = y * mp->avctx->width;

    int w, i, x = 0;



    p = mp->vpt[y];

    if (mp->changes_map[y0 + x] == 0) {

        memset(mp->gradient_scale, 1, sizeof(mp->gradient_scale));

        ++x;

    }

    while (x < mp->avctx->width) {

        w = mp->changes_map[y0 + x];

        if (w != 0) {

            if ((y & 3) == 0) {

                if (mp->changes_map[y0 + x + mp->avctx->width] < w ||

                    mp->changes_map[y0 + x + mp->avctx->width * 2] < w ||

                    mp->changes_map[y0 + x + mp->avctx->width * 3] < w) {

                    for (i = (x + 3) & ~3; i < x + w; i += 4) {

                        mp->hpt[((y / 4) * mp->avctx->width + i) / 4] = mp_get_yuv_from_rgb(mp, i, y);

                    }

                }

            }

            x += w;

            memset(mp->gradient_scale, 1, sizeof(mp->gradient_scale));

            p = mp_get_yuv_from_rgb(mp, x - 1, y);

        } else {

            p.y += mp_gradient(mp, 0, mp_get_vlc(mp, gb));


            if ((x & 3) == 0) {

                if ((y & 3) == 0) {

                    p.v += mp_gradient(mp, 1, mp_get_vlc(mp, gb));


                    p.u += mp_gradient(mp, 2, mp_get_vlc(mp, gb));


                    mp->hpt[((y / 4) * mp->avctx->width + x) / 4] = p;

                } else {

                    p.v = mp->hpt[((y / 4) * mp->avctx->width + x) / 4].v;

                    p.u = mp->hpt[((y / 4) * mp->avctx->width + x) / 4].u;

                }

            }

            mp_set_rgb_from_yuv(mp, x, y, &p);

            ++x;

        }

    }

}