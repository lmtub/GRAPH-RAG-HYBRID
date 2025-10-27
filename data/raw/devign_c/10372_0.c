static gboolean gd_leave_event(GtkWidget *widget, GdkEventCrossing *crossing,

                               gpointer opaque)

{

    VirtualConsole *vc = opaque;

    GtkDisplayState *s = vc->s;



    if (!gd_is_grab_active(s) && gd_grab_on_hover(s)) {

        gd_ungrab_keyboard(s);

    }



    return TRUE;

}
