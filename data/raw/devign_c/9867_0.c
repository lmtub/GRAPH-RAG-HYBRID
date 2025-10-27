void mmio_ide_init (target_phys_addr_t membase, target_phys_addr_t membase2,

                    MemoryRegion *address_space,

                    qemu_irq irq, int shift,

                    DriveInfo *hd0, DriveInfo *hd1)

{

    MMIOState *s = g_malloc0(sizeof(MMIOState));



    ide_init2_with_non_qdev_drives(&s->bus, hd0, hd1, irq);



    s->shift = shift;



    memory_region_init_io(&s->iomem1, &mmio_ide_ops, s,

                          "ide-mmio.1", 16 << shift);

    memory_region_init_io(&s->iomem2, &mmio_ide_cs_ops, s,

                          "ide-mmio.2", 2 << shift);

    memory_region_add_subregion(address_space, membase, &s->iomem1);

    memory_region_add_subregion(address_space, membase2, &s->iomem2);

    vmstate_register(NULL, 0, &vmstate_ide_mmio, s);

    qemu_register_reset(mmio_ide_reset, s);

}
