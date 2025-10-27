static void gd_set_keycode_type(GtkDisplayState *s)
{
#ifdef GDK_WINDOWING_X11
    GdkDisplay *display = gtk_widget_get_display(s->window);
    if (GDK_IS_X11_DISPLAY(display)) {
        Display *x11_display = gdk_x11_display_get_xdisplay(display);
        XkbDescPtr desc = XkbGetKeyboard(x11_display, XkbGBN_AllComponentsMask,
                                         XkbUseCoreKbd);
        char *keycodes = NULL;
        if (desc && desc->names) {
            keycodes = XGetAtomName(x11_display, desc->names->keycodes);
        if (keycodes == NULL) {
            fprintf(stderr, "could not lookup keycode name\n");
        } else if (strstart(keycodes, "evdev", NULL)) {
            s->has_evdev = true;
        } else if (!strstart(keycodes, "xfree86", NULL)) {
            fprintf(stderr, "unknown keycodes `%s', please report to "
                    "qemu-devel@nongnu.org\n", keycodes);
#endif