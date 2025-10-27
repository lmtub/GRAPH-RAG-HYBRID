static void qxl_render_update_area_unlocked(PCIQXLDevice *qxl)

{

    VGACommonState *vga = &qxl->vga;

    DisplaySurface *surface;

    int i;



    if (qxl->guest_primary.resized) {

        qxl->guest_primary.resized = 0;

        qxl->guest_primary.data = qxl_phys2virt(qxl,

                                                qxl->guest_primary.surface.mem,

                                                MEMSLOT_GROUP_GUEST);

        if (!qxl->guest_primary.data) {

            return;

        }

        qxl_set_rect_to_surface(qxl, &qxl->dirty[0]);

        qxl->num_dirty_rects = 1;

        trace_qxl_render_guest_primary_resized(

               qxl->guest_primary.surface.width,

               qxl->guest_primary.surface.height,

               qxl->guest_primary.qxl_stride,

               qxl->guest_primary.bytes_pp,

               qxl->guest_primary.bits_pp);

        if (qxl->guest_primary.qxl_stride > 0) {

            surface = qemu_create_displaysurface_from

                (qxl->guest_primary.surface.width,

                 qxl->guest_primary.surface.height,

                 qxl->guest_primary.bits_pp,

                 qxl->guest_primary.abs_stride,

                 qxl->guest_primary.data,

                 false);

        } else {

            surface = qemu_create_displaysurface

                (qxl->guest_primary.surface.width,

                 qxl->guest_primary.surface.height);

        }

        dpy_gfx_replace_surface(vga->con, surface);

    }



    if (!qxl->guest_primary.data) {

        return;

    }

    for (i = 0; i < qxl->num_dirty_rects; i++) {

        if (qemu_spice_rect_is_empty(qxl->dirty+i)) {

            break;

        }

        if (qxl->dirty[i].left > qxl->dirty[i].right ||

            qxl->dirty[i].top > qxl->dirty[i].bottom ||

            qxl->dirty[i].right > qxl->guest_primary.surface.width ||

            qxl->dirty[i].bottom > qxl->guest_primary.surface.height) {

            continue;

        }

        qxl_blit(qxl, qxl->dirty+i);

        dpy_gfx_update(vga->con,

                       qxl->dirty[i].left, qxl->dirty[i].top,

                       qxl->dirty[i].right - qxl->dirty[i].left,

                       qxl->dirty[i].bottom - qxl->dirty[i].top);

    }

    qxl->num_dirty_rects = 0;

}
