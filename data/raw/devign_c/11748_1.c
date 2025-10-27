static int esp_pci_scsi_init(PCIDevice *dev)

{

    PCIESPState *pci = PCI_ESP(dev);

    DeviceState *d = DEVICE(dev);

    ESPState *s = &pci->esp;

    uint8_t *pci_conf;

    Error *err = NULL;



    pci_conf = dev->config;



    /* Interrupt pin A */

    pci_conf[PCI_INTERRUPT_PIN] = 0x01;



    s->dma_memory_read = esp_pci_dma_memory_read;

    s->dma_memory_write = esp_pci_dma_memory_write;

    s->dma_opaque = pci;

    s->chip_id = TCHI_AM53C974;

    memory_region_init_io(&pci->io, OBJECT(pci), &esp_pci_io_ops, pci,

                          "esp-io", 0x80);



    pci_register_bar(dev, 0, PCI_BASE_ADDRESS_SPACE_IO, &pci->io);

    s->irq = pci_allocate_irq(dev);



    scsi_bus_new(&s->bus, sizeof(s->bus), d, &esp_pci_scsi_info, NULL);

    if (!d->hotplugged) {

        scsi_bus_legacy_handle_cmdline(&s->bus, &err);

        if (err != NULL) {


            error_free(err);

            return -1;

        }

    }

    return 0;

}