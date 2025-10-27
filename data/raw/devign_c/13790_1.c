static void vmsvga_init(DeviceState *dev, struct vmsvga_state_s *s,

                        MemoryRegion *address_space, MemoryRegion *io)

{

    s->scratch_size = SVGA_SCRATCH_SIZE;

    s->scratch = g_malloc(s->scratch_size * 4);



    s->vga.con = graphic_console_init(dev, 0, &vmsvga_ops, s);



    s->fifo_size = SVGA_FIFO_SIZE;

    memory_region_init_ram(&s->fifo_ram, NULL, "vmsvga.fifo", s->fifo_size,

                           &error_abort);

    vmstate_register_ram_global(&s->fifo_ram);

    s->fifo_ptr = memory_region_get_ram_ptr(&s->fifo_ram);



    vga_common_init(&s->vga, OBJECT(dev), true);

    vga_init(&s->vga, OBJECT(dev), address_space, io, true);

    vmstate_register(NULL, 0, &vmstate_vga_common, &s->vga);

    s->new_depth = 32;

}
