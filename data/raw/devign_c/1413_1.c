void vga_common_init(VGACommonState *s, Object *obj, bool global_vmstate)

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



    s->vram_size_mb = uint_clamp(s->vram_size_mb, 1, 512);

    s->vram_size_mb = pow2ceil(s->vram_size_mb);

    s->vram_size = s->vram_size_mb << 20;



    if (!s->vbe_size) {

        s->vbe_size = s->vram_size;

    }



    s->is_vbe_vmstate = 1;

    memory_region_init_ram(&s->vram, obj, "vga.vram", s->vram_size,

                           &error_abort);

    vmstate_register_ram(&s->vram, global_vmstate ? NULL : DEVICE(obj));

    xen_register_framebuffer(&s->vram);

    s->vram_ptr = memory_region_get_ram_ptr(&s->vram);

    s->get_bpp = vga_get_bpp;

    s->get_offsets = vga_get_offsets;

    s->get_resolution = vga_get_resolution;

    s->hw_ops = &vga_ops;

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



    /*

     * Set default fb endian based on target, could probably be turned

     * into a device attribute set by the machine/platform to remove

     * all target endian dependencies from this file.

     */

#ifdef TARGET_WORDS_BIGENDIAN

    s->default_endian_fb = true;

#else

    s->default_endian_fb = false;

#endif

    vga_dirty_log_start(s);

}
