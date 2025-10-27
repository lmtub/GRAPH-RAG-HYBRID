static MaltaFPGAState *malta_fpga_init(MemoryRegion *address_space,

         target_phys_addr_t base, qemu_irq uart_irq, CharDriverState *uart_chr)

{

    MaltaFPGAState *s;



    s = (MaltaFPGAState *)g_malloc0(sizeof(MaltaFPGAState));



    memory_region_init_io(&s->iomem, &malta_fpga_ops, s,

                          "malta-fpga", 0x100000);

    memory_region_init_alias(&s->iomem_lo, "malta-fpga",

                             &s->iomem, 0, 0x900);

    memory_region_init_alias(&s->iomem_hi, "malta-fpga",

                             &s->iomem, 0xa00, 0x10000-0xa00);



    memory_region_add_subregion(address_space, base, &s->iomem_lo);

    memory_region_add_subregion(address_space, base + 0xa00, &s->iomem_hi);



    s->display = qemu_chr_new("fpga", "vc:320x200", malta_fpga_led_init);



    s->uart = serial_mm_init(address_space, base + 0x900, 3, uart_irq,

                             230400, uart_chr, DEVICE_NATIVE_ENDIAN);



    malta_fpga_reset(s);

    qemu_register_reset(malta_fpga_reset, s);



    return s;

}
