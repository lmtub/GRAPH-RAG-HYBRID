static void gd_resize(DisplayChangeListener *dcl,

                      DisplayState *ds)

{

    GtkDisplayState *s = ds->opaque;

    cairo_format_t kind;

    int stride;



    DPRINTF("resize(width=%d, height=%d)\n",

            ds_get_width(ds), ds_get_height(ds));



    if (s->surface) {

        cairo_surface_destroy(s->surface);

    }



    switch (ds->surface->pf.bits_per_pixel) {

    case 8:

        kind = CAIRO_FORMAT_A8;

        break;

    case 16:

        kind = CAIRO_FORMAT_RGB16_565;

        break;

    case 32:

        kind = CAIRO_FORMAT_RGB24;

        break;

    default:

        g_assert_not_reached();

        break;

    }



    stride = cairo_format_stride_for_width(kind, ds_get_width(ds));

    g_assert(ds_get_linesize(ds) == stride);



    s->surface = cairo_image_surface_create_for_data(ds_get_data(ds),

                                                     kind,

                                                     ds_get_width(ds),

                                                     ds_get_height(ds),

                                                     ds_get_linesize(ds));



    if (!s->full_screen) {

        GtkRequisition req;

        double sx, sy;



        if (s->free_scale) {

            sx = s->scale_x;

            sy = s->scale_y;



            s->scale_y = 1.0;

            s->scale_x = 1.0;

        } else {

            sx = 1.0;

            sy = 1.0;

        }



        gtk_widget_set_size_request(s->drawing_area,

                                    ds_get_width(ds) * s->scale_x,

                                    ds_get_height(ds) * s->scale_y);

#if GTK_CHECK_VERSION(3, 0, 0)

        gtk_widget_get_preferred_size(s->vbox, NULL, &req);

#else

        gtk_widget_size_request(s->vbox, &req);

#endif



        gtk_window_resize(GTK_WINDOW(s->window),

                          req.width * sx, req.height * sy);

    }

}
