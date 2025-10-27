static void virtio_init_pci(VirtIOPCIProxy *proxy, VirtIODevice *vdev,

                            uint16_t vendor, uint16_t device,

                            uint16_t class_code, uint8_t pif)

{

    uint8_t *config;

    uint32_t size;



    proxy->vdev = vdev;



    config = proxy->pci_dev.config;

    pci_config_set_vendor_id(config, vendor);

    pci_config_set_device_id(config, device);



    config[0x08] = VIRTIO_PCI_ABI_VERSION;



    config[0x09] = pif;

    pci_config_set_class(config, class_code);

    config[PCI_HEADER_TYPE] = PCI_HEADER_TYPE_NORMAL;



    config[0x2c] = vendor & 0xFF;

    config[0x2d] = (vendor >> 8) & 0xFF;

    config[0x2e] = vdev->device_id & 0xFF;

    config[0x2f] = (vdev->device_id >> 8) & 0xFF;



    config[0x3d] = 1;



    if (vdev->nvectors && !msix_init(&proxy->pci_dev, vdev->nvectors, 1, 0,

                                     TARGET_PAGE_SIZE)) {

        pci_register_bar(&proxy->pci_dev, 1,

                         msix_bar_size(&proxy->pci_dev),

                         PCI_ADDRESS_SPACE_MEM,

                         msix_mmio_map);

    } else

        vdev->nvectors = 0;



    proxy->pci_dev.config_write = virtio_write_config;



    size = VIRTIO_PCI_REGION_SIZE(&proxy->pci_dev) + vdev->config_len;

    if (size & (size-1))

        size = 1 << qemu_fls(size);



    pci_register_bar(&proxy->pci_dev, 0, size, PCI_ADDRESS_SPACE_IO,

                           virtio_map);



    qemu_register_reset(virtio_pci_reset, proxy);



    virtio_bind_device(vdev, &virtio_pci_bindings, proxy);

}
