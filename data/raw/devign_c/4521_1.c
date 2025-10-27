static gboolean gd_motion_event(GtkWidget *widget, GdkEventMotion *motion,

                                void *opaque)

{

    GtkDisplayState *s = opaque;

    int x, y;

    int mx, my;

    int fbh, fbw;

    int ww, wh;



    fbw = surface_width(s->ds) * s->scale_x;

    fbh = surface_height(s->ds) * s->scale_y;



    gdk_drawable_get_size(gtk_widget_get_window(s->drawing_area), &ww, &wh);



    mx = my = 0;

    if (ww > fbw) {

        mx = (ww - fbw) / 2;

    }

    if (wh > fbh) {

        my = (wh - fbh) / 2;

    }



    x = (motion->x - mx) / s->scale_x;

    y = (motion->y - my) / s->scale_y;



    if (x < 0 || y < 0 ||

        x >= surface_width(s->ds) ||

        y >= surface_height(s->ds)) {

        return TRUE;

    }



    if (qemu_input_is_absolute()) {

        qemu_input_queue_abs(s->dcl.con, INPUT_AXIS_X, x,

                             surface_width(s->ds));

        qemu_input_queue_abs(s->dcl.con, INPUT_AXIS_Y, y,

                             surface_height(s->ds));

        qemu_input_event_sync();

    } else if (s->last_x != -1 && s->last_y != -1 && gd_is_grab_active(s)) {

        qemu_input_queue_rel(s->dcl.con, INPUT_AXIS_X, x - s->last_x);

        qemu_input_queue_rel(s->dcl.con, INPUT_AXIS_Y, y - s->last_y);

        qemu_input_event_sync();

    }

    s->last_x = x;

    s->last_y = y;



    if (!qemu_input_is_absolute() && gd_is_grab_active(s)) {

        GdkScreen *screen = gtk_widget_get_screen(s->drawing_area);

        int x = (int)motion->x_root;

        int y = (int)motion->y_root;



        /* In relative mode check to see if client pointer hit

         * one of the screen edges, and if so move it back by

         * 200 pixels. This is important because the pointer

         * in the server doesn't correspond 1-for-1, and so

         * may still be only half way across the screen. Without

         * this warp, the server pointer would thus appear to hit

         * an invisible wall */

        if (x == 0) {

            x += 200;

        }

        if (y == 0) {

            y += 200;

        }

        if (x == (gdk_screen_get_width(screen) - 1)) {

            x -= 200;

        }

        if (y == (gdk_screen_get_height(screen) - 1)) {

            y -= 200;

        }



        if (x != (int)motion->x_root || y != (int)motion->y_root) {

#if GTK_CHECK_VERSION(3, 0, 0)

            GdkDevice *dev = gdk_event_get_device((GdkEvent *)motion);

            gdk_device_warp(dev, screen, x, y);

#else

            GdkDisplay *display = gtk_widget_get_display(widget);

            gdk_display_warp_pointer(display, screen, x, y);

#endif

            s->last_x = -1;

            s->last_y = -1;

            return FALSE;

        }

    }

    return TRUE;

}
