static int milkymist_softusb_init(SysBusDevice *dev)

{

    MilkymistSoftUsbState *s = MILKYMIST_SOFTUSB(dev);



    sysbus_init_irq(dev, &s->irq);



    memory_region_init_io(&s->regs_region, OBJECT(s), &softusb_mmio_ops, s,

                          "milkymist-softusb", R_MAX * 4);

    sysbus_init_mmio(dev, &s->regs_region);



    /* register pmem and dmem */

    memory_region_init_ram(&s->pmem, OBJECT(s), "milkymist-softusb.pmem",

                           s->pmem_size, &error_abort);

    vmstate_register_ram_global(&s->pmem);

    s->pmem_ptr = memory_region_get_ram_ptr(&s->pmem);

    sysbus_init_mmio(dev, &s->pmem);

    memory_region_init_ram(&s->dmem, OBJECT(s), "milkymist-softusb.dmem",

                           s->dmem_size, &error_abort);

    vmstate_register_ram_global(&s->dmem);

    s->dmem_ptr = memory_region_get_ram_ptr(&s->dmem);

    sysbus_init_mmio(dev, &s->dmem);



    hid_init(&s->hid_kbd, HID_KEYBOARD, softusb_kbd_hid_datain);

    hid_init(&s->hid_mouse, HID_MOUSE, softusb_mouse_hid_datain);



    return 0;

}
