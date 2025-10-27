static void xenfb_guest_copy(struct XenFB *xenfb, int x, int y, int w, int h)

{

    DisplaySurface *surface = qemu_console_surface(xenfb->c.con);

    int line, oops = 0;

    int bpp = surface_bits_per_pixel(surface);

    int linesize = surface_stride(surface);

    uint8_t *data = surface_data(surface);



    if (!is_buffer_shared(surface)) {

        switch (xenfb->depth) {

        case 8:

            if (bpp == 16) {

                BLT(uint8_t, uint16_t,   3, 3, 2,   5, 6, 5);

            } else if (bpp == 32) {

                BLT(uint8_t, uint32_t,   3, 3, 2,   8, 8, 8);

            } else {

                oops = 1;

            }

            break;

        case 24:

            if (bpp == 16) {

                BLT(uint32_t, uint16_t,  8, 8, 8,   5, 6, 5);

            } else if (bpp == 32) {

                BLT(uint32_t, uint32_t,  8, 8, 8,   8, 8, 8);

            } else {

                oops = 1;

            }

            break;

        default:

            oops = 1;

	}

    }

    if (oops) /* should not happen */

        xen_pv_printf(&xenfb->c.xendev, 0, "%s: oops: convert %d -> %d bpp?\n",

                      __FUNCTION__, xenfb->depth, bpp);



    dpy_gfx_update(xenfb->c.con, x, y, w, h);

}
