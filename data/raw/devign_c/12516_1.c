static void exynos4210_fimd_update(void *opaque)

{

    Exynos4210fimdState *s = (Exynos4210fimdState *)opaque;

    DisplaySurface *surface = qemu_console_surface(s->console);

    Exynos4210fimdWindow *w;

    int i, line;

    hwaddr fb_line_addr, inc_size;

    int scrn_height;

    int first_line = -1, last_line = -1, scrn_width;

    bool blend = false;

    uint8_t *host_fb_addr;

    bool is_dirty = false;

    const int global_width = (s->vidtcon[2] & FIMD_VIDTCON2_SIZE_MASK) + 1;

    const int global_height = ((s->vidtcon[2] >> FIMD_VIDTCON2_VER_SHIFT) &

            FIMD_VIDTCON2_SIZE_MASK) + 1;



    if (!s || !s->console || !surface_bits_per_pixel(surface) ||

            !s->enabled) {

        return;

    }

    exynos4210_update_resolution(s);



    for (i = 0; i < NUM_OF_WINDOWS; i++) {

        w = &s->window[i];

        if ((w->wincon & FIMD_WINCON_ENWIN) && w->host_fb_addr) {

            scrn_height = w->rightbot_y - w->lefttop_y + 1;

            scrn_width = w->virtpage_width;

            /* Total width of virtual screen page in bytes */

            inc_size = scrn_width + w->virtpage_offsize;

            memory_region_sync_dirty_bitmap(w->mem_section.mr);

            host_fb_addr = w->host_fb_addr;

            fb_line_addr = w->mem_section.offset_within_region;



            for (line = 0; line < scrn_height; line++) {

                is_dirty = memory_region_get_dirty(w->mem_section.mr,

                            fb_line_addr, scrn_width, DIRTY_MEMORY_VGA);



                if (s->invalidate || is_dirty) {

                    if (first_line == -1) {

                        first_line = line;

                    }

                    last_line = line;

                    w->draw_line(w, host_fb_addr, s->ifb +

                        w->lefttop_x * RGBA_SIZE + (w->lefttop_y + line) *

                        global_width * RGBA_SIZE, blend);

                }

                host_fb_addr += inc_size;

                fb_line_addr += inc_size;

                is_dirty = false;

            }

            memory_region_reset_dirty(w->mem_section.mr,

                w->mem_section.offset_within_region,

                w->fb_len, DIRTY_MEMORY_VGA);

            blend = true;

        }

    }



    /* Copy resulting image to QEMU_CONSOLE. */

    if (first_line >= 0) {

        uint8_t *d;

        int bpp;



        bpp = surface_bits_per_pixel(surface);

        fimd_update_putpix_qemu(bpp);

        bpp = (bpp + 1) >> 3;

        d = surface_data(surface);

        for (line = first_line; line <= last_line; line++) {

            fimd_copy_line_toqemu(global_width, s->ifb + global_width * line *

                    RGBA_SIZE, d + global_width * line * bpp);

        }

        dpy_gfx_update(s->console, 0, 0, global_width, global_height);

    }

    s->invalidate = false;

    s->vidintcon[1] |= FIMD_VIDINT_INTFRMPEND;

    if ((s->vidcon[0] & FIMD_VIDCON0_ENVID_F) == 0) {

        exynos4210_fimd_enable(s, false);

    }

    exynos4210_fimd_update_irq(s);

}
