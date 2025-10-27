static void gd_update_caption(GtkDisplayState *s)

{

    const char *status = "";

    gchar *title;



    if (!runstate_is_running()) {

        status = " [Stopped]";

    }



    if (qemu_name) {

        title = g_strdup_printf("QEMU (%s)%s", qemu_name, status);

    } else {

        title = g_strdup_printf("QEMU%s", status);

    }



    gtk_window_set_title(GTK_WINDOW(s->window), title);



    g_free(title);

}
