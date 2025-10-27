static void pxa2xx_lcdc_dma0_redraw_rot270(PXA2xxLCDState *s,

               hwaddr addr, int *miny, int *maxy)

{

    DisplaySurface *surface = qemu_console_surface(s->con);

    int src_width, dest_width;

    drawfn fn = NULL;

    if (s->dest_width) {

        fn = s->line_fn[s->transp][s->bpp];

    }

    if (!fn) {

        return;

    }



    src_width = (s->xres + 3) & ~3;     /* Pad to a 4 pixels multiple */

    if (s->bpp == pxa_lcdc_19pbpp || s->bpp == pxa_lcdc_18pbpp) {

        src_width *= 3;

    } else if (s->bpp > pxa_lcdc_16bpp) {

        src_width *= 4;

    } else if (s->bpp > pxa_lcdc_8bpp) {

        src_width *= 2;

    }



    dest_width = s->yres * s->dest_width;

    *miny = 0;

    framebuffer_update_display(surface, s->sysmem,

                               addr, s->xres, s->yres,

                               src_width, -s->dest_width, dest_width,

                               s->invalidated,

                               fn, s->dma_ch[0].palette,

                               miny, maxy);

}
