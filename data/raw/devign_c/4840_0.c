static void vgafb_update_display(void *opaque)

{

    MilkymistVgafbState *s = opaque;

    SysBusDevice *sbd;

    DisplaySurface *surface = qemu_console_surface(s->con);

    int first = 0;

    int last = 0;

    drawfn fn;



    if (!vgafb_enabled(s)) {

        return;

    }



    sbd = SYS_BUS_DEVICE(s);

    int dest_width = s->regs[R_HRES];



    switch (surface_bits_per_pixel(surface)) {

    case 0:

        return;

    case 8:

        fn = draw_line_8;

        break;

    case 15:

        fn = draw_line_15;

        dest_width *= 2;

        break;

    case 16:

        fn = draw_line_16;

        dest_width *= 2;

        break;

    case 24:

        fn = draw_line_24;

        dest_width *= 3;

        break;

    case 32:

        fn = draw_line_32;

        dest_width *= 4;

        break;

    default:

        hw_error("milkymist_vgafb: bad color depth\n");

        break;

    }



    framebuffer_update_display(surface, sysbus_address_space(sbd),

                               s->regs[R_BASEADDRESS] + s->fb_offset,

                               s->regs[R_HRES],

                               s->regs[R_VRES],

                               s->regs[R_HRES] * 2,

                               dest_width,

                               0,

                               s->invalidate,

                               fn,

                               NULL,

                               &first, &last);



    if (first >= 0) {

        dpy_gfx_update(s->con, 0, first, s->regs[R_HRES], last - first + 1);

    }

    s->invalidate = 0;

}
