static void gd_ungrab_pointer(GtkDisplayState *s)

{

#if GTK_CHECK_VERSION(3, 0, 0)

    GdkDisplay *display = gtk_widget_get_display(s->drawing_area);

    GdkDeviceManager *mgr = gdk_display_get_device_manager(display);

    GList *devices = gdk_device_manager_list_devices(mgr,

                                                     GDK_DEVICE_TYPE_MASTER);

    GList *tmp = devices;

    while (tmp) {

        GdkDevice *dev = tmp->data;

        if (gdk_device_get_source(dev) == GDK_SOURCE_MOUSE) {

            gdk_device_ungrab(dev,

                              GDK_CURRENT_TIME);

        }

        tmp = tmp->next;

    }

    g_list_free(devices);

#else

    gdk_pointer_ungrab(GDK_CURRENT_TIME);

#endif

}
