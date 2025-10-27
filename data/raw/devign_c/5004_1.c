static int afx_init1(SysBusDevice *dev)

{

    AFXState *s = TCX_AFX(dev);



    memory_region_init_ram(&s->mem, OBJECT(s), "sun4m.afx", 4, &error_abort);

    vmstate_register_ram_global(&s->mem);

    sysbus_init_mmio(dev, &s->mem);

    return 0;

}
