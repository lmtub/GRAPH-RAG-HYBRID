static void mp_decode_frame_helper(MotionPixelsContext *mp, GetBitContext *gb)
{
    YuvPixel p;
    int y, y0;
    for (y = 0; y < mp->avctx->height; ++y) {
        if (mp->changes_map[y * mp->avctx->width] != 0) {
            memset(mp->gradient_scale, 1, sizeof(mp->gradient_scale));
            p = mp_get_yuv_from_rgb(mp, 0, y);
        } else {
            p.y += mp_gradient(mp, 0, mp_get_vlc(mp, gb));
            p.y = av_clip(p.y, 0, 31);
            if ((y & 3) == 0) {
                p.v += mp_gradient(mp, 1, mp_get_vlc(mp, gb));
                p.v = av_clip(p.v, -32, 31);
                p.u += mp_gradient(mp, 2, mp_get_vlc(mp, gb));
                p.u = av_clip(p.u, -32, 31);
            }
            mp->vpt[y] = p;
            mp_set_rgb_from_yuv(mp, 0, y, &p);
        }
    }
    for (y0 = 0; y0 < 2; ++y0)
        for (y = y0; y < mp->avctx->height; y += 2)
            mp_decode_line(mp, gb, y);
}