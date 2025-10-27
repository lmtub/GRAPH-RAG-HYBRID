static void vnc_resize(VncState *vs)

{

    DisplayState *ds = vs->ds;



    int size_changed;



    vs->old_data = qemu_realloc(vs->old_data, ds_get_linesize(ds) * ds_get_height(ds));



    if (vs->old_data == NULL) {

        fprintf(stderr, "vnc: memory allocation failed\n");

        exit(1);

    }



    if (ds_get_bytes_per_pixel(ds) != vs->serverds.pf.bytes_per_pixel)

        console_color_init(ds);

    vnc_colordepth(vs);

    size_changed = ds_get_width(ds) != vs->serverds.width ||

                   ds_get_height(ds) != vs->serverds.height;

    vs->serverds = *(ds->surface);

    if (size_changed) {

        if (vs->csock != -1 && vnc_has_feature(vs, VNC_FEATURE_RESIZE)) {

            vnc_write_u8(vs, 0);  /* msg id */

            vnc_write_u8(vs, 0);

            vnc_write_u16(vs, 1); /* number of rects */

            vnc_framebuffer_update(vs, 0, 0, ds_get_width(ds), ds_get_height(ds),

                                   VNC_ENCODING_DESKTOPRESIZE);

            vnc_flush(vs);

        }

    }



    memset(vs->dirty_row, 0xFF, sizeof(vs->dirty_row));

    memset(vs->old_data, 42, ds_get_linesize(vs->ds) * ds_get_height(vs->ds));

}
