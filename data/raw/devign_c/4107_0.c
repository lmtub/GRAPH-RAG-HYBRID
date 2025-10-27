void vga_common_init(VGACommonState *s)

{

    int i, j, v, b;



    for(i = 0;i < 256; i++) {

        v = 0;

        for(j = 0; j < 8; j++) {

            v |= ((i >> j) & 1) << (j * 4);

        }

        expand4[i] = v;



        v = 0;

        for(j = 0; j < 4; j++) {

            v |= ((i >> (2 * j)) & 3) << (j * 4);

        }

        expand2[i] = v;

    }

    for(i = 0; i < 16; i++) {

        v = 0;

        for(j = 0; j < 4; j++) {

            b = ((i >> j) & 1);

            v |= b << (2 * j);

            v |= b << (2 * j + 1);

        }

        expand4to8[i] = v;

    }



    /* valid range: 1 MB -> 256 MB */

    s->vram_size = 1024 * 1024;

    while (s->vram_size < (s->vram_size_mb << 20) &&

           s->vram_size < (256 << 20)) {

        s->vram_size <<= 1;

    }

    s->vram_size_mb = s->vram_size >> 20;



    s->is_vbe_vmstate = 1;

    memory_region_init_ram(&s->vram, "vga.vram", s->vram_size);

    vmstate_register_ram_global(&s->vram);

    xen_register_framebuffer(&s->vram);

    s->vram_ptr = memory_region_get_ram_ptr(&s->vram);

    s->get_bpp = vga_get_bpp;

    s->get_offsets = vga_get_offsets;

    s->get_resolution = vga_get_resolution;

    s->update = vga_update_display;

    s->invalidate = vga_invalidate_display;

    s->screen_dump = vga_screen_dump;

    s->text_update = vga_update_text;

    switch (vga_retrace_method) {

    case VGA_RETRACE_DUMB:

        s->retrace = vga_dumb_retrace;

        s->update_retrace_info = vga_dumb_update_retrace_info;

        break;



    case VGA_RETRACE_PRECISE:

        s->retrace = vga_precise_retrace;

        s->update_retrace_info = vga_precise_update_retrace_info;

        break;

    }

    vga_dirty_log_start(s);

}
