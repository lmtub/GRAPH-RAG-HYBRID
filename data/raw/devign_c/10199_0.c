static void vnc_dpy_resize(DisplayState *ds)

{

    int size_changed;

    VncDisplay *vd = ds->opaque;

    VncState *vs;



    /* server surface */

    if (!vd->server)

        vd->server = qemu_mallocz(sizeof(*vd->server));

    if (vd->server->data)

        qemu_free(vd->server->data);

    *(vd->server) = *(ds->surface);

    vd->server->data = qemu_mallocz(vd->server->linesize *

                                    vd->server->height);



    /* guest surface */

    if (!vd->guest.ds)

        vd->guest.ds = qemu_mallocz(sizeof(*vd->guest.ds));

    if (ds_get_bytes_per_pixel(ds) != vd->guest.ds->pf.bytes_per_pixel)

        console_color_init(ds);

    size_changed = ds_get_width(ds) != vd->guest.ds->width ||

                   ds_get_height(ds) != vd->guest.ds->height;

    *(vd->guest.ds) = *(ds->surface);

    memset(vd->guest.dirty, 0xFF, sizeof(vd->guest.dirty));



    QTAILQ_FOREACH(vs, &vd->clients, next) {

        vnc_colordepth(vs);

        if (size_changed) {

            vnc_desktop_resize(vs);

        }

        if (vs->vd->cursor) {

            vnc_cursor_define(vs);

        }

        memset(vs->dirty, 0xFF, sizeof(vs->dirty));

    }

}
