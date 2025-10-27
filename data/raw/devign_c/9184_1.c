static void vga_isa_realizefn(DeviceState *dev, Error **errp)

{

    ISADevice *isadev = ISA_DEVICE(dev);

    ISAVGAState *d = ISA_VGA(dev);

    VGACommonState *s = &d->state;

    MemoryRegion *vga_io_memory;

    const MemoryRegionPortio *vga_ports, *vbe_ports;



    vga_common_init(s, OBJECT(dev), true);

    s->legacy_address_space = isa_address_space(isadev);

    vga_io_memory = vga_init_io(s, OBJECT(dev), &vga_ports, &vbe_ports);

    isa_register_portio_list(isadev, 0x3b0, vga_ports, s, "vga");

    if (vbe_ports) {

        isa_register_portio_list(isadev, 0x1ce, vbe_ports, s, "vbe");

    }

    memory_region_add_subregion_overlap(isa_address_space(isadev),

                                        0x000a0000,

                                        vga_io_memory, 1);

    memory_region_set_coalescing(vga_io_memory);

    s->con = graphic_console_init(DEVICE(dev), 0, s->hw_ops, s);



    vga_init_vbe(s, OBJECT(dev), isa_address_space(isadev));

    /* ROM BIOS */

    rom_add_vga(VGABIOS_FILENAME);

}
