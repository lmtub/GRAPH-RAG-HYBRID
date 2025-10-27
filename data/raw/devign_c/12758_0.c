static int pl181_init(SysBusDevice *sbd)

{

    DeviceState *dev = DEVICE(sbd);

    PL181State *s = PL181(dev);

    DriveInfo *dinfo;



    memory_region_init_io(&s->iomem, OBJECT(s), &pl181_ops, s, "pl181", 0x1000);

    sysbus_init_mmio(sbd, &s->iomem);

    sysbus_init_irq(sbd, &s->irq[0]);

    sysbus_init_irq(sbd, &s->irq[1]);

    qdev_init_gpio_out(dev, s->cardstatus, 2);

    dinfo = drive_get_next(IF_SD);

    s->card = sd_init(dinfo ? blk_bs(blk_by_legacy_dinfo(dinfo)) : NULL, false);

    if (s->card == NULL) {

        return -1;

    }



    return 0;

}
