static GtkWidget *gd_create_menu_machine(GtkDisplayState *s, GtkAccelGroup *accel_group)

{

    GtkWidget *machine_menu;

    GtkWidget *separator;



    machine_menu = gtk_menu_new();

    gtk_menu_set_accel_group(GTK_MENU(machine_menu), accel_group);



    s->pause_item = gtk_check_menu_item_new_with_mnemonic(_("_Pause"));

    gtk_menu_shell_append(GTK_MENU_SHELL(machine_menu), s->pause_item);



    separator = gtk_separator_menu_item_new();

    gtk_menu_shell_append(GTK_MENU_SHELL(machine_menu), separator);



    s->reset_item = gtk_menu_item_new_with_mnemonic(_("_Reset"));

    gtk_menu_shell_append(GTK_MENU_SHELL(machine_menu), s->reset_item);



    s->powerdown_item = gtk_menu_item_new_with_mnemonic(_("Power _Down"));

    gtk_menu_shell_append(GTK_MENU_SHELL(machine_menu), s->powerdown_item);



    separator = gtk_separator_menu_item_new();

    gtk_menu_shell_append(GTK_MENU_SHELL(machine_menu), separator);



    s->quit_item = gtk_menu_item_new_with_mnemonic(_("_Quit"));

    gtk_menu_item_set_accel_path(GTK_MENU_ITEM(s->quit_item),

                                 "<QEMU>/Machine/Quit");

    gtk_accel_map_add_entry("<QEMU>/Machine/Quit",

                            GDK_KEY_q, GDK_CONTROL_MASK);

    gtk_menu_shell_append(GTK_MENU_SHELL(machine_menu), s->quit_item);



    return machine_menu;

}
