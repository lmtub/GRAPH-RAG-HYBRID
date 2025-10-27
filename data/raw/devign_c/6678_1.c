static void vnc_dpy_setdata(DisplayChangeListener *dcl,

                            DisplayState *ds)

{

    VncDisplay *vd = ds->opaque;



    qemu_pixman_image_unref(vd->guest.fb);

    vd->guest.fb = pixman_image_ref(ds->surface->image);

    vd->guest.format = ds->surface->format;

    vnc_dpy_update(dcl, ds, 0, 0, ds_get_width(ds), ds_get_height(ds));

}
