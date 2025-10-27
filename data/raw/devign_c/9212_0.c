static void gd_connect_signals(GtkDisplayState *s)

{

    g_signal_connect(s->show_tabs_item, "activate",

                     G_CALLBACK(gd_menu_show_tabs), s);



    g_signal_connect(s->window, "key-press-event",

                     G_CALLBACK(gd_window_key_event), s);

    g_signal_connect(s->window, "delete-event",

                     G_CALLBACK(gd_window_close), s);



#if GTK_CHECK_VERSION(3, 0, 0)

    g_signal_connect(s->drawing_area, "draw",

                     G_CALLBACK(gd_draw_event), s);

#else

    g_signal_connect(s->drawing_area, "expose-event",

                     G_CALLBACK(gd_expose_event), s);

#endif

    g_signal_connect(s->drawing_area, "motion-notify-event",

                     G_CALLBACK(gd_motion_event), s);

    g_signal_connect(s->drawing_area, "button-press-event",

                     G_CALLBACK(gd_button_event), s);

    g_signal_connect(s->drawing_area, "button-release-event",

                     G_CALLBACK(gd_button_event), s);

    g_signal_connect(s->drawing_area, "scroll-event",

                     G_CALLBACK(gd_scroll_event), s);

    g_signal_connect(s->drawing_area, "key-press-event",

                     G_CALLBACK(gd_key_event), s);

    g_signal_connect(s->drawing_area, "key-release-event",

                     G_CALLBACK(gd_key_event), s);



    g_signal_connect(s->pause_item, "activate",

                     G_CALLBACK(gd_menu_pause), s);

    g_signal_connect(s->reset_item, "activate",

                     G_CALLBACK(gd_menu_reset), s);

    g_signal_connect(s->powerdown_item, "activate",

                     G_CALLBACK(gd_menu_powerdown), s);

    g_signal_connect(s->quit_item, "activate",

                     G_CALLBACK(gd_menu_quit), s);

    g_signal_connect(s->full_screen_item, "activate",

                     G_CALLBACK(gd_menu_full_screen), s);

    g_signal_connect(s->zoom_in_item, "activate",

                     G_CALLBACK(gd_menu_zoom_in), s);

    g_signal_connect(s->zoom_out_item, "activate",

                     G_CALLBACK(gd_menu_zoom_out), s);

    g_signal_connect(s->zoom_fixed_item, "activate",

                     G_CALLBACK(gd_menu_zoom_fixed), s);

    g_signal_connect(s->zoom_fit_item, "activate",

                     G_CALLBACK(gd_menu_zoom_fit), s);

    g_signal_connect(s->vga_item, "activate",

                     G_CALLBACK(gd_menu_switch_vc), s);

    g_signal_connect(s->grab_item, "activate",

                     G_CALLBACK(gd_menu_grab_input), s);

    g_signal_connect(s->notebook, "switch-page",

                     G_CALLBACK(gd_change_page), s);

    g_signal_connect(s->drawing_area, "enter-notify-event",

                     G_CALLBACK(gd_enter_event), s);

    g_signal_connect(s->drawing_area, "leave-notify-event",

                     G_CALLBACK(gd_leave_event), s);

    g_signal_connect(s->drawing_area, "focus-out-event",

                     G_CALLBACK(gd_focus_out_event), s);

}
