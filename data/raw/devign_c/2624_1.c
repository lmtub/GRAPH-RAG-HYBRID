static int idreg_init1(SysBusDevice *dev)

{

    IDRegState *s = MACIO_ID_REGISTER(dev);



    memory_region_init_ram(&s->mem, OBJECT(s),

                           "sun4m.idreg", sizeof(idreg_data), &error_abort);

    vmstate_register_ram_global(&s->mem);

    memory_region_set_readonly(&s->mem, true);

    sysbus_init_mmio(dev, &s->mem);

    return 0;

}
