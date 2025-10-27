int vnc_tight_send_framebuffer_update(VncState *vs, int x, int y,

                                      int w, int h)

{

    int max_rows;



    if (vs->clientds.pf.bytes_per_pixel == 4 && vs->clientds.pf.rmax == 0xFF &&

        vs->clientds.pf.bmax == 0xFF && vs->clientds.pf.gmax == 0xFF) {

        vs->tight_pixel24 = true;

    } else {

        vs->tight_pixel24 = false;

    }



    if (w * h < VNC_TIGHT_MIN_SPLIT_RECT_SIZE)

        return send_rect_simple(vs, x, y, w, h);



    /* Calculate maximum number of rows in one non-solid rectangle. */



    max_rows = tight_conf[vs->tight_compression].max_rect_size;

    max_rows /= MIN(tight_conf[vs->tight_compression].max_rect_width, w);



    return find_large_solid_color_rect(vs, x, y, w, h, max_rows);

}
