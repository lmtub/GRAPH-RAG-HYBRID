static int intel_hda_init(PCIDevice *pci)

{

    IntelHDAState *d = DO_UPCAST(IntelHDAState, pci, pci);

    uint8_t *conf = d->pci.config;



    d->name = d->pci.qdev.info->name;



    pci_config_set_vendor_id(conf, PCI_VENDOR_ID_INTEL);

    pci_config_set_device_id(conf, 0x2668);

    pci_config_set_revision(conf, 1);

    pci_config_set_class(conf, PCI_CLASS_MULTIMEDIA_HD_AUDIO);

    pci_config_set_interrupt_pin(conf, 1);



    /* HDCTL off 0x40 bit 0 selects signaling mode (1-HDA, 0 - Ac97) 18.1.19 */

    conf[0x40] = 0x01;



    d->mmio_addr = cpu_register_io_memory(intel_hda_mmio_read,

                                          intel_hda_mmio_write, d,

                                          DEVICE_NATIVE_ENDIAN);

    pci_register_bar_simple(&d->pci, 0, 0x4000, 0, d->mmio_addr);

    if (d->msi) {

        msi_init(&d->pci, 0x50, 1, true, false);

    }



    hda_codec_bus_init(&d->pci.qdev, &d->codecs,

                       intel_hda_response, intel_hda_xfer);



    return 0;

}
