PCIBus *i440fx_init(PCII440FXState **pi440fx_state, int *piix3_devfn, qemu_irq *pic)

{

    DeviceState *dev;

    PCIBus *b;

    PCIDevice *d;

    I440FXState *s;

    PIIX3State *piix3;



    dev = qdev_create(NULL, "i440FX-pcihost");

    s = FROM_SYSBUS(I440FXState, sysbus_from_qdev(dev));

    b = pci_bus_new(&s->busdev.qdev, NULL, 0);

    s->bus = b;

    qdev_init(dev);



    d = pci_create_simple(b, 0, "i440FX");

    *pi440fx_state = DO_UPCAST(PCII440FXState, dev, d);



    piix3 = DO_UPCAST(PIIX3State, dev,

                                 pci_create_simple(b, -1, "PIIX3"));

    piix3->pic = pic;

    pci_bus_irqs(b, piix3_set_irq, pci_slot_get_pirq, piix3, 4);

    (*pi440fx_state)->piix3 = piix3;



    *piix3_devfn = piix3->dev.devfn;



    return b;

}
