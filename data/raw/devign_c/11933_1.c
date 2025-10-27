static void piix3_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);



    dc->desc        = "ISA bridge";

    dc->vmsd        = &vmstate_piix3;

    dc->no_user     = 1,

    k->no_hotplug   = 1;

    k->init         = piix3_initfn;

    k->config_write = piix3_write_config;

    k->vendor_id    = PCI_VENDOR_ID_INTEL;

    /* 82371SB PIIX3 PCI-to-ISA bridge (Step A1) */

    k->device_id    = PCI_DEVICE_ID_INTEL_82371SB_0;

    k->class_id     = PCI_CLASS_BRIDGE_ISA;

}
