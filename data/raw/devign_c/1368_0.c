static void gd_mouse_mode_change(Notifier *notify, void *data)

{

    gd_update_cursor(container_of(notify, GtkDisplayState, mouse_mode_notifier),

                     FALSE);

}
