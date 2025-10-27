void gtk_display_init(DisplayState *ds, bool full_screen, bool grab_on_hover)

{

    GtkDisplayState *s = g_malloc0(sizeof(*s));

    char *filename;



    gtk_init(NULL, NULL);



    s->window = gtk_window_new(GTK_WINDOW_TOPLEVEL);

#if GTK_CHECK_VERSION(3, 2, 0)

    s->vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);

#else

    s->vbox = gtk_vbox_new(FALSE, 0);

#endif

    s->notebook = gtk_notebook_new();

    s->menu_bar = gtk_menu_bar_new();



    s->free_scale = FALSE;



    setlocale(LC_ALL, "");

    bindtextdomain("qemu", CONFIG_QEMU_LOCALEDIR);

    textdomain("qemu");



    s->null_cursor = gdk_cursor_new(GDK_BLANK_CURSOR);



    s->mouse_mode_notifier.notify = gd_mouse_mode_change;

    qemu_add_mouse_mode_change_notifier(&s->mouse_mode_notifier);

    qemu_add_vm_change_state_handler(gd_change_runstate, s);



    filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, "qemu_logo_no_text.svg");

    if (filename) {

        GError *error = NULL;

        GdkPixbuf *pixbuf = gdk_pixbuf_new_from_file(filename, &error);

        if (pixbuf) {

            gtk_window_set_icon(GTK_WINDOW(s->window), pixbuf);

        } else {

            g_error_free(error);

        }

        g_free(filename);

    }



    gd_create_menus(s);



    gd_connect_signals(s);



    gtk_notebook_set_show_tabs(GTK_NOTEBOOK(s->notebook), FALSE);

    gtk_notebook_set_show_border(GTK_NOTEBOOK(s->notebook), FALSE);



    gd_update_caption(s);



    gtk_box_pack_start(GTK_BOX(s->vbox), s->menu_bar, FALSE, TRUE, 0);

    gtk_box_pack_start(GTK_BOX(s->vbox), s->notebook, TRUE, TRUE, 0);



    gtk_container_add(GTK_CONTAINER(s->window), s->vbox);



    gtk_widget_show_all(s->window);



#ifdef VTE_RESIZE_HACK

    {

        VirtualConsole *cur = gd_vc_find_current(s);

        int i;



        for (i = 0; i < s->nb_vcs; i++) {

            VirtualConsole *vc = &s->vc[i];

            if (vc && vc->type == GD_VC_VTE && vc != cur) {

                gtk_widget_hide(vc->vte.terminal);

            }

        }

        gd_update_windowsize(cur);

    }

#endif



    if (full_screen) {

        gtk_menu_item_activate(GTK_MENU_ITEM(s->full_screen_item));

    }

    if (grab_on_hover) {

        gtk_menu_item_activate(GTK_MENU_ITEM(s->grab_on_hover_item));

    }



    gd_set_keycode_type(s);

}
