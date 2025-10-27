static void sdhci_sysbus_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->vmsd = &sdhci_vmstate;

    dc->props = sdhci_sysbus_properties;

    dc->realize = sdhci_sysbus_realize;

    dc->reset = sdhci_poweron_reset;






}