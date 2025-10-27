static int ram_init1(SysBusDevice *dev)

{

    RamDevice *d = SUN4U_RAM(dev);



    memory_region_init_ram(&d->ram, OBJECT(d), "sun4u.ram", d->size,

                           &error_abort);

    vmstate_register_ram_global(&d->ram);

    sysbus_init_mmio(dev, &d->ram);

    return 0;

}
