static void lsi_scsi_init(PCIDevice *dev)
{
    LSIState *s = (LSIState *)dev;
    uint8_t *pci_conf;
    pci_conf = s->pci_dev.config;
    /* PCI Vendor ID (word) */
    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_LSI_LOGIC);
    /* PCI device ID (word) */
    pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_LSI_53C895A);
    /* PCI base class code */
    pci_config_set_class(pci_conf, PCI_CLASS_STORAGE_SCSI);
    /* PCI subsystem ID */
    pci_conf[0x2e] = 0x00;
    pci_conf[0x2f] = 0x10;
    /* PCI latency timer = 255 */
    pci_conf[0x0d] = 0xff;
    /* Interrupt pin 1 */
    pci_conf[0x3d] = 0x01;
    s->mmio_io_addr = cpu_register_io_memory(lsi_mmio_readfn,
                                             lsi_mmio_writefn, s);
    s->ram_io_addr = cpu_register_io_memory(lsi_ram_readfn,
                                            lsi_ram_writefn, s);
    pci_register_bar((struct PCIDevice *)s, 0, 256,
                           PCI_ADDRESS_SPACE_IO, lsi_io_mapfunc);
    pci_register_bar((struct PCIDevice *)s, 1, 0x400,
                           PCI_ADDRESS_SPACE_MEM, lsi_mmio_mapfunc);
    pci_register_bar((struct PCIDevice *)s, 2, 0x2000,
                           PCI_ADDRESS_SPACE_MEM, lsi_ram_mapfunc);
    s->queue = qemu_malloc(sizeof(lsi_queue));
    s->queue_len = 1;
    s->active_commands = 0;
    s->pci_dev.unregister = lsi_scsi_uninit;
    lsi_soft_reset(s);
    scsi_bus_new(&dev->qdev, lsi_scsi_attach);
}