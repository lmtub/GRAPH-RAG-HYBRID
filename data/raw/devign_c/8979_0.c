static GtkWidget *gd_create_menu_view(GtkDisplayState *s, GtkAccelGroup *accel_group)

{

    GSList *group = NULL;

    GtkWidget *view_menu;

    GtkWidget *separator;

    int i;



    view_menu = gtk_menu_new();

    gtk_menu_set_accel_group(GTK_MENU(view_menu), accel_group);



    s->full_screen_item =

        gtk_image_menu_item_new_from_stock(GTK_STOCK_FULLSCREEN, NULL);

    gtk_menu_item_set_accel_path(GTK_MENU_ITEM(s->full_screen_item),

                                 "<QEMU>/View/Full Screen");

    gtk_accel_map_add_entry("<QEMU>/View/Full Screen", GDK_KEY_f, GDK_CONTROL_MASK | GDK_MOD1_MASK);

    gtk_menu_shell_append(GTK_MENU_SHELL(view_menu), s->full_screen_item);



    separator = gtk_separator_menu_item_new();

    gtk_menu_shell_append(GTK_MENU_SHELL(view_menu), separator);



    s->zoom_in_item = gtk_image_menu_item_new_from_stock(GTK_STOCK_ZOOM_IN, NULL);

    gtk_menu_item_set_accel_path(GTK_MENU_ITEM(s->zoom_in_item),

                                 "<QEMU>/View/Zoom In");

    gtk_accel_map_add_entry("<QEMU>/View/Zoom In", GDK_KEY_plus, GDK_CONTROL_MASK | GDK_MOD1_MASK);

    gtk_menu_shell_append(GTK_MENU_SHELL(view_menu), s->zoom_in_item);



    s->zoom_out_item = gtk_image_menu_item_new_from_stock(GTK_STOCK_ZOOM_OUT, NULL);

    gtk_menu_item_set_accel_path(GTK_MENU_ITEM(s->zoom_out_item),

                                 "<QEMU>/View/Zoom Out");

    gtk_accel_map_add_entry("<QEMU>/View/Zoom Out", GDK_KEY_minus, GDK_CONTROL_MASK | GDK_MOD1_MASK);

    gtk_menu_shell_append(GTK_MENU_SHELL(view_menu), s->zoom_out_item);



    s->zoom_fixed_item = gtk_image_menu_item_new_from_stock(GTK_STOCK_ZOOM_100, NULL);

    gtk_menu_item_set_accel_path(GTK_MENU_ITEM(s->zoom_fixed_item),

                                 "<QEMU>/View/Zoom Fixed");

    gtk_accel_map_add_entry("<QEMU>/View/Zoom Fixed", GDK_KEY_0, GDK_CONTROL_MASK | GDK_MOD1_MASK);

    gtk_menu_shell_append(GTK_MENU_SHELL(view_menu), s->zoom_fixed_item);



    s->zoom_fit_item = gtk_check_menu_item_new_with_mnemonic(_("Zoom To _Fit"));

    gtk_menu_shell_append(GTK_MENU_SHELL(view_menu), s->zoom_fit_item);



    separator = gtk_separator_menu_item_new();

    gtk_menu_shell_append(GTK_MENU_SHELL(view_menu), separator);



    s->grab_on_hover_item = gtk_check_menu_item_new_with_mnemonic(_("Grab On _Hover"));

    gtk_menu_shell_append(GTK_MENU_SHELL(view_menu), s->grab_on_hover_item);



    s->grab_item = gtk_check_menu_item_new_with_mnemonic(_("_Grab Input"));

    gtk_menu_item_set_accel_path(GTK_MENU_ITEM(s->grab_item),

                                 "<QEMU>/View/Grab Input");

    gtk_accel_map_add_entry("<QEMU>/View/Grab Input", GDK_KEY_g, GDK_CONTROL_MASK | GDK_MOD1_MASK);

    gtk_menu_shell_append(GTK_MENU_SHELL(view_menu), s->grab_item);



    separator = gtk_separator_menu_item_new();

    gtk_menu_shell_append(GTK_MENU_SHELL(view_menu), separator);



    s->vga_item = gtk_radio_menu_item_new_with_mnemonic(group, "_VGA");

    group = gtk_radio_menu_item_get_group(GTK_RADIO_MENU_ITEM(s->vga_item));

    gtk_menu_item_set_accel_path(GTK_MENU_ITEM(s->vga_item),

                                 "<QEMU>/View/VGA");

    gtk_accel_map_add_entry("<QEMU>/View/VGA", GDK_KEY_1, GDK_CONTROL_MASK | GDK_MOD1_MASK);

    gtk_menu_shell_append(GTK_MENU_SHELL(view_menu), s->vga_item);



    for (i = 0; i < nb_vcs; i++) {

        VirtualConsole *vc = &s->vc[i];



        group = gd_vc_init(s, vc, i, group, view_menu);

        s->nb_vcs++;

    }



    separator = gtk_separator_menu_item_new();

    gtk_menu_shell_append(GTK_MENU_SHELL(view_menu), separator);



    s->show_tabs_item = gtk_check_menu_item_new_with_mnemonic(_("Show _Tabs"));

    gtk_menu_shell_append(GTK_MENU_SHELL(view_menu), s->show_tabs_item);



    return view_menu;

}
