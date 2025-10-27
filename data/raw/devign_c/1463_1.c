static GSList *gd_vc_init(GtkDisplayState *s, VirtualConsole *vc, int index, GSList *group,

                          GtkWidget *view_menu)

{

#if defined(CONFIG_VTE)

    const char *label;

    char buffer[32];

    char path[32];

#if VTE_CHECK_VERSION(0, 26, 0)

    VtePty *pty;

#endif

    GIOChannel *chan;

    GtkWidget *scrolled_window;

    GtkAdjustment *vadjustment;

    int master_fd, slave_fd;



    snprintf(buffer, sizeof(buffer), "vc%d", index);

    snprintf(path, sizeof(path), "<QEMU>/View/VC%d", index);



    vc->chr = vcs[index];



    if (vc->chr->label) {

        label = vc->chr->label;

    } else {

        label = buffer;

    }



    vc->menu_item = gtk_radio_menu_item_new_with_mnemonic(group, label);

    group = gtk_radio_menu_item_get_group(GTK_RADIO_MENU_ITEM(vc->menu_item));

    gtk_menu_item_set_accel_path(GTK_MENU_ITEM(vc->menu_item), path);

    gtk_accel_map_add_entry(path, GDK_KEY_2 + index, HOTKEY_MODIFIERS);



    vc->terminal = vte_terminal_new();



    master_fd = qemu_openpty_raw(&slave_fd, NULL);

    g_assert(master_fd != -1);



#if VTE_CHECK_VERSION(0, 26, 0)

    pty = vte_pty_new_foreign(master_fd, NULL);

    vte_terminal_set_pty_object(VTE_TERMINAL(vc->terminal), pty);

#else

    vte_terminal_set_pty(VTE_TERMINAL(vc->terminal), master_fd);

#endif



    vte_terminal_set_scrollback_lines(VTE_TERMINAL(vc->terminal), -1);



#if VTE_CHECK_VERSION(0, 28, 0) && GTK_CHECK_VERSION(3, 0, 0)

    vadjustment = gtk_scrollable_get_vadjustment(GTK_SCROLLABLE(vc->terminal));

#else

    vadjustment = vte_terminal_get_adjustment(VTE_TERMINAL(vc->terminal));

#endif



    scrolled_window = gtk_scrolled_window_new(NULL, vadjustment);

    gtk_container_add(GTK_CONTAINER(scrolled_window), vc->terminal);



    vte_terminal_set_size(VTE_TERMINAL(vc->terminal), 80, 25);



    vc->fd = slave_fd;

    vc->chr->opaque = vc;

    vc->scrolled_window = scrolled_window;



    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(vc->scrolled_window),

                                   GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);



    gtk_notebook_append_page(GTK_NOTEBOOK(s->notebook), scrolled_window, gtk_label_new(label));

    g_signal_connect(vc->menu_item, "activate",

                     G_CALLBACK(gd_menu_switch_vc), s);



    gtk_menu_shell_append(GTK_MENU_SHELL(view_menu), vc->menu_item);



    qemu_chr_be_generic_open(vc->chr);

    if (vc->chr->init) {

        vc->chr->init(vc->chr);

    }



    chan = g_io_channel_unix_new(vc->fd);

    g_io_add_watch(chan, G_IO_IN, gd_vc_in, vc);



#endif /* CONFIG_VTE */

    return group;

}
