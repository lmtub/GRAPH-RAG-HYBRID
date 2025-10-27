static int send_sub_rect(VncState *vs, int x, int y, int w, int h)

{

    VncPalette *palette = &color_count_palette;

    uint32_t bg = 0, fg = 0;

    int colors;

    int ret = 0;

#ifdef CONFIG_VNC_JPEG

    bool force_jpeg = false;

    bool allow_jpeg = true;

#endif



    vnc_framebuffer_update(vs, x, y, w, h, vs->tight.type);



    vnc_tight_start(vs);

    vnc_raw_send_framebuffer_update(vs, x, y, w, h);

    vnc_tight_stop(vs);



#ifdef CONFIG_VNC_JPEG

    if (!vs->vd->non_adaptive && vs->tight.quality != (uint8_t)-1) {

        double freq = vnc_update_freq(vs, x, y, w, h);



        if (freq < tight_jpeg_conf[vs->tight.quality].jpeg_freq_min) {

            allow_jpeg = false;

        }

        if (freq >= tight_jpeg_conf[vs->tight.quality].jpeg_freq_threshold) {

            force_jpeg = true;

            vnc_sent_lossy_rect(vs, x, y, w, h);

        }

    }

#endif



    colors = tight_fill_palette(vs, x, y, w * h, &bg, &fg, palette);



#ifdef CONFIG_VNC_JPEG

    if (allow_jpeg && vs->tight.quality != (uint8_t)-1) {

        ret = send_sub_rect_jpeg(vs, x, y, w, h, bg, fg, colors, palette,

                                 force_jpeg);

    } else {

        ret = send_sub_rect_nojpeg(vs, x, y, w, h, bg, fg, colors, palette);

    }

#else

    ret = send_sub_rect_nojpeg(vs, x, y, w, h, bg, fg, colors, palette);

#endif



    return ret;

}
