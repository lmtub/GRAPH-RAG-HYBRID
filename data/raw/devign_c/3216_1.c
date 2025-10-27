static void vmsvga_init(struct vmsvga_state_s *s, int vga_ram_size)

{

    s->scratch_size = SVGA_SCRATCH_SIZE;

    s->scratch = qemu_malloc(s->scratch_size * 4);



    vmsvga_reset(s);



    s->fifo_size = SVGA_FIFO_SIZE;

    s->fifo_offset = qemu_ram_alloc(s->fifo_size);

    s->fifo_ptr = qemu_get_ram_ptr(s->fifo_offset);



    vga_common_init(&s->vga, vga_ram_size);

    vga_init(&s->vga);

    vmstate_register(0, &vmstate_vga_common, &s->vga);



    s->vga.ds = graphic_console_init(vmsvga_update_display,

                                     vmsvga_invalidate_display,

                                     vmsvga_screen_dump,

                                     vmsvga_text_update, s);



    vga_init_vbe(&s->vga);

    rom_add_vga(VGABIOS_FILENAME);

}
