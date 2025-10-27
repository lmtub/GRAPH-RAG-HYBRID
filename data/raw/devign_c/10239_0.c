static void g364fb_init(DeviceState *dev, G364State *s)

{

    s->vram = g_malloc0(s->vram_size);



    s->con = graphic_console_init(g364fb_update_display,

                                  g364fb_invalidate_display,

                                  g364fb_screen_dump, NULL, s);



    memory_region_init_io(&s->mem_ctrl, &g364fb_ctrl_ops, s, "ctrl", 0x180000);

    memory_region_init_ram_ptr(&s->mem_vram, "vram",

                               s->vram_size, s->vram);

    vmstate_register_ram(&s->mem_vram, dev);

    memory_region_set_coalescing(&s->mem_vram);

}
