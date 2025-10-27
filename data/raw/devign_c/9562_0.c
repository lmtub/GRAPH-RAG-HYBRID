static void fimd_update_memory_section(Exynos4210fimdState *s, unsigned win)

{

    Exynos4210fimdWindow *w = &s->window[win];

    target_phys_addr_t fb_start_addr, fb_mapped_len;



    if (!s->enabled || !(w->wincon & FIMD_WINCON_ENWIN) ||

            FIMD_WINDOW_PROTECTED(s->shadowcon, win)) {

        return;

    }



    if (w->host_fb_addr) {

        cpu_physical_memory_unmap(w->host_fb_addr, w->fb_len, 0, 0);

        w->host_fb_addr = NULL;

        w->fb_len = 0;

    }



    fb_start_addr = w->buf_start[fimd_get_buffer_id(w)];

    /* Total number of bytes of virtual screen used by current window */

    w->fb_len = fb_mapped_len = (w->virtpage_width + w->virtpage_offsize) *

            (w->rightbot_y - w->lefttop_y + 1);

    w->mem_section = memory_region_find(sysbus_address_space(&s->busdev),

            fb_start_addr, w->fb_len);

    assert(w->mem_section.mr);

    assert(w->mem_section.offset_within_address_space == fb_start_addr);

    DPRINT_TRACE("Window %u framebuffer changed: address=0x%08x, len=0x%x\n",

            win, fb_start_addr, w->fb_len);



    if (w->mem_section.size != w->fb_len ||

            !memory_region_is_ram(w->mem_section.mr)) {

        DPRINT_ERROR("Failed to find window %u framebuffer region\n", win);

        goto error_return;

    }



    w->host_fb_addr = cpu_physical_memory_map(fb_start_addr, &fb_mapped_len, 0);

    if (!w->host_fb_addr) {

        DPRINT_ERROR("Failed to map window %u framebuffer\n", win);

        goto error_return;

    }



    if (fb_mapped_len != w->fb_len) {

        DPRINT_ERROR("Window %u mapped framebuffer length is less then "

                "expected\n", win);

        cpu_physical_memory_unmap(w->host_fb_addr, fb_mapped_len, 0, 0);

        goto error_return;

    }

    return;



error_return:

    w->mem_section.mr = NULL;

    w->mem_section.size = 0;

    w->host_fb_addr = NULL;

    w->fb_len = 0;

}
