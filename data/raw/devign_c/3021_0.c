int isa_vga_mm_init(hwaddr vram_base,

                    hwaddr ctrl_base, int it_shift,

                    MemoryRegion *address_space)

{

    ISAVGAMMState *s;



    s = g_malloc0(sizeof(*s));



    s->vga.vram_size_mb = VGA_RAM_SIZE >> 20;

    vga_common_init(&s->vga);

    vga_mm_init(s, vram_base, ctrl_base, it_shift, address_space);



    s->vga.con = graphic_console_init(s->vga.update, s->vga.invalidate,

                                      s->vga.screen_dump, s->vga.text_update,

                                      s);



    vga_init_vbe(&s->vga, address_space);

    return 0;

}
