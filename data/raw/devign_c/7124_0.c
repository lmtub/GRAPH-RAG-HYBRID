ISADevice *rtc_init(ISABus *bus, int base_year, qemu_irq intercept_irq)

{

    DeviceState *dev;

    ISADevice *isadev;

    RTCState *s;



    isadev = isa_create(bus, TYPE_MC146818_RTC);

    dev = DEVICE(isadev);

    s = MC146818_RTC(isadev);

    qdev_prop_set_int32(dev, "base_year", base_year);

    qdev_init_nofail(dev);

    if (intercept_irq) {

        s->irq = intercept_irq;

    } else {

        isa_init_irq(isadev, &s->irq, RTC_ISA_IRQ);

    }

    QLIST_INSERT_HEAD(&rtc_devices, s, link);



    return isadev;

}
