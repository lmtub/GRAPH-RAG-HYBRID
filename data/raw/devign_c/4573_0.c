static void omap_screen_dump(void *opaque, const char *filename, bool cswitch,

                             Error **errp)

{

    struct omap_lcd_panel_s *omap_lcd = opaque;

    DisplaySurface *surface = qemu_console_surface(omap_lcd->con);



    omap_update_display(opaque);

    if (omap_lcd && surface_data(surface))

        omap_ppm_save(filename, surface_data(surface),

                    omap_lcd->width, omap_lcd->height,

                    surface_stride(surface), errp);

}
