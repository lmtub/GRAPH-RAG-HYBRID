static int send_sub_rect_solid(VncState *vs, int x, int y, int w, int h)

{

    vnc_framebuffer_update(vs, x, y, w, h, VNC_ENCODING_TIGHT);



    vnc_tight_start(vs);

    vnc_raw_send_framebuffer_update(vs, x, y, w, h);

    vnc_tight_stop(vs);



    return send_solid_rect(vs);

}
