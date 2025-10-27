static void gd_update_geometry_hints(VirtualConsole *vc)

{

    GtkDisplayState *s = vc->s;

    GdkWindowHints mask = 0;

    GdkGeometry geo = {};

    GtkWidget *geo_widget = NULL;

    GtkWindow *geo_window;



    if (vc->type == GD_VC_GFX) {

        if (!vc->gfx.ds) {

            return;

        }

        if (s->free_scale) {

            geo.min_width  = surface_width(vc->gfx.ds) * VC_SCALE_MIN;

            geo.min_height = surface_height(vc->gfx.ds) * VC_SCALE_MIN;

            mask |= GDK_HINT_MIN_SIZE;

        } else {

            geo.min_width  = surface_width(vc->gfx.ds) * vc->gfx.scale_x;

            geo.min_height = surface_height(vc->gfx.ds) * vc->gfx.scale_y;

            mask |= GDK_HINT_MIN_SIZE;

        }

        geo_widget = vc->gfx.drawing_area;

        gtk_widget_set_size_request(geo_widget, geo.min_width, geo.min_height);



#if defined(CONFIG_VTE)

    } else if (vc->type == GD_VC_VTE) {

        VteTerminal *term = VTE_TERMINAL(vc->vte.terminal);

        GtkBorder *ib;



        geo.width_inc  = vte_terminal_get_char_width(term);

        geo.height_inc = vte_terminal_get_char_height(term);

        mask |= GDK_HINT_RESIZE_INC;

        geo.base_width  = geo.width_inc;

        geo.base_height = geo.height_inc;

        mask |= GDK_HINT_BASE_SIZE;

        geo.min_width  = geo.width_inc * VC_TERM_X_MIN;

        geo.min_height = geo.height_inc * VC_TERM_Y_MIN;

        mask |= GDK_HINT_MIN_SIZE;

        gtk_widget_style_get(vc->vte.terminal, "inner-border", &ib, NULL);

        geo.base_width  += ib->left + ib->right;

        geo.base_height += ib->top + ib->bottom;

        geo.min_width   += ib->left + ib->right;

        geo.min_height  += ib->top + ib->bottom;

        geo_widget = vc->vte.terminal;

#endif

    }



    geo_window = GTK_WINDOW(vc->window ? vc->window : s->window);

    gtk_window_set_geometry_hints(geo_window, geo_widget, &geo, mask);

}
