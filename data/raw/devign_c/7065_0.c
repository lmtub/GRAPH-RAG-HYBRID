void qxl_render_update(PCIQXLDevice *qxl)

{

    VGACommonState *vga = &qxl->vga;

    QXLRect dirty[32], update;

    void *ptr;

    int i;



    if (qxl->guest_primary.resized) {

        qxl->guest_primary.resized = 0;



        if (qxl->guest_primary.flipped) {

            g_free(qxl->guest_primary.flipped);

            qxl->guest_primary.flipped = NULL;

        }

        qemu_free_displaysurface(vga->ds);



        qxl->guest_primary.data = memory_region_get_ram_ptr(&qxl->vga.vram);

        if (qxl->guest_primary.stride < 0) {

            /* spice surface is upside down -> need extra buffer to flip */

            qxl->guest_primary.stride = -qxl->guest_primary.stride;

            qxl->guest_primary.flipped = g_malloc(qxl->guest_primary.surface.width *

                                                     qxl->guest_primary.stride);

            ptr = qxl->guest_primary.flipped;

        } else {

            ptr = qxl->guest_primary.data;

        }

        dprint(qxl, 1, "%s: %dx%d, stride %d, bpp %d, depth %d, flip %s\n",

               __FUNCTION__,

               qxl->guest_primary.surface.width,

               qxl->guest_primary.surface.height,

               qxl->guest_primary.stride,

               qxl->guest_primary.bytes_pp,

               qxl->guest_primary.bits_pp,

               qxl->guest_primary.flipped ? "yes" : "no");

        vga->ds->surface =

            qemu_create_displaysurface_from(qxl->guest_primary.surface.width,

                                            qxl->guest_primary.surface.height,

                                            qxl->guest_primary.bits_pp,

                                            qxl->guest_primary.stride,

                                            ptr);

        dpy_resize(vga->ds);

    }



    if (!qxl->guest_primary.commands) {

        return;

    }

    qxl->guest_primary.commands = 0;



    update.left   = 0;

    update.right  = qxl->guest_primary.surface.width;

    update.top    = 0;

    update.bottom = qxl->guest_primary.surface.height;



    memset(dirty, 0, sizeof(dirty));

    qxl_spice_update_area(qxl, 0, &update,

                          dirty, ARRAY_SIZE(dirty), 1, QXL_SYNC);



    for (i = 0; i < ARRAY_SIZE(dirty); i++) {

        if (qemu_spice_rect_is_empty(dirty+i)) {

            break;

        }

        if (qxl->guest_primary.flipped) {

            qxl_flip(qxl, dirty+i);

        }

        dpy_update(vga->ds,

                   dirty[i].left, dirty[i].top,

                   dirty[i].right - dirty[i].left,

                   dirty[i].bottom - dirty[i].top);

    }

}
