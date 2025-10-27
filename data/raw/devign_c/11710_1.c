static void virtio_pci_device_plugged(DeviceState *d, Error **errp)

{

    VirtIOPCIProxy *proxy = VIRTIO_PCI(d);

    VirtioBusState *bus = &proxy->bus;

    bool legacy = virtio_pci_legacy(proxy);

    bool modern;

    bool modern_pio = proxy->flags & VIRTIO_PCI_FLAG_MODERN_PIO_NOTIFY;

    uint8_t *config;

    uint32_t size;

    VirtIODevice *vdev = virtio_bus_get_device(&proxy->bus);



    /*

     * Virtio capabilities present without

     * VIRTIO_F_VERSION_1 confuses guests

     */

    if (!proxy->ignore_backend_features &&

            !virtio_has_feature(vdev->host_features, VIRTIO_F_VERSION_1)) {

        virtio_pci_disable_modern(proxy);



        if (!legacy) {

            error_setg(errp, "Device doesn't support modern mode, and legacy"

                             " mode is disabled");

            error_append_hint(errp, "Set disable-legacy to off\n");



            return;

        }

    }



    modern = virtio_pci_modern(proxy);



    config = proxy->pci_dev.config;

    if (proxy->class_code) {

        pci_config_set_class(config, proxy->class_code);

    }



    if (legacy) {

        if (virtio_host_has_feature(vdev, VIRTIO_F_IOMMU_PLATFORM)) {

            error_setg(errp, "VIRTIO_F_IOMMU_PLATFORM was supported by"

                       "neither legacy nor transitional device.");

            return ;

        }

        /* legacy and transitional */

        pci_set_word(config + PCI_SUBSYSTEM_VENDOR_ID,

                     pci_get_word(config + PCI_VENDOR_ID));

        pci_set_word(config + PCI_SUBSYSTEM_ID, virtio_bus_get_vdev_id(bus));

    } else {

        /* pure virtio-1.0 */

        pci_set_word(config + PCI_VENDOR_ID,

                     PCI_VENDOR_ID_REDHAT_QUMRANET);

        pci_set_word(config + PCI_DEVICE_ID,

                     0x1040 + virtio_bus_get_vdev_id(bus));

        pci_config_set_revision(config, 1);

    }

    config[PCI_INTERRUPT_PIN] = 1;





    if (modern) {

        struct virtio_pci_cap cap = {

            .cap_len = sizeof cap,

        };

        struct virtio_pci_notify_cap notify = {

            .cap.cap_len = sizeof notify,

            .notify_off_multiplier =

                cpu_to_le32(virtio_pci_queue_mem_mult(proxy)),

        };

        struct virtio_pci_cfg_cap cfg = {

            .cap.cap_len = sizeof cfg,

            .cap.cfg_type = VIRTIO_PCI_CAP_PCI_CFG,

        };

        struct virtio_pci_notify_cap notify_pio = {

            .cap.cap_len = sizeof notify,

            .notify_off_multiplier = cpu_to_le32(0x0),

        };



        struct virtio_pci_cfg_cap *cfg_mask;



        virtio_pci_modern_regions_init(proxy);



        virtio_pci_modern_mem_region_map(proxy, &proxy->common, &cap);

        virtio_pci_modern_mem_region_map(proxy, &proxy->isr, &cap);

        virtio_pci_modern_mem_region_map(proxy, &proxy->device, &cap);

        virtio_pci_modern_mem_region_map(proxy, &proxy->notify, &notify.cap);



        if (modern_pio) {

            memory_region_init(&proxy->io_bar, OBJECT(proxy),

                               "virtio-pci-io", 0x4);



            pci_register_bar(&proxy->pci_dev, proxy->modern_io_bar_idx,

                             PCI_BASE_ADDRESS_SPACE_IO, &proxy->io_bar);



            virtio_pci_modern_io_region_map(proxy, &proxy->notify_pio,

                                            &notify_pio.cap);

        }



        pci_register_bar(&proxy->pci_dev, proxy->modern_mem_bar_idx,

                         PCI_BASE_ADDRESS_SPACE_MEMORY |

                         PCI_BASE_ADDRESS_MEM_PREFETCH |

                         PCI_BASE_ADDRESS_MEM_TYPE_64,

                         &proxy->modern_bar);



        proxy->config_cap = virtio_pci_add_mem_cap(proxy, &cfg.cap);

        cfg_mask = (void *)(proxy->pci_dev.wmask + proxy->config_cap);

        pci_set_byte(&cfg_mask->cap.bar, ~0x0);

        pci_set_long((uint8_t *)&cfg_mask->cap.offset, ~0x0);

        pci_set_long((uint8_t *)&cfg_mask->cap.length, ~0x0);

        pci_set_long(cfg_mask->pci_cfg_data, ~0x0);

    }



    if (proxy->nvectors) {

        int err = msix_init_exclusive_bar(&proxy->pci_dev, proxy->nvectors,

                                          proxy->msix_bar_idx);

        if (err) {

            /* Notice when a system that supports MSIx can't initialize it.  */

            if (err != -ENOTSUP) {

                error_report("unable to init msix vectors to %" PRIu32,

                             proxy->nvectors);

            }

            proxy->nvectors = 0;

        }

    }



    proxy->pci_dev.config_write = virtio_write_config;

    proxy->pci_dev.config_read = virtio_read_config;



    if (legacy) {

        size = VIRTIO_PCI_REGION_SIZE(&proxy->pci_dev)

            + virtio_bus_get_vdev_config_len(bus);

        size = pow2ceil(size);



        memory_region_init_io(&proxy->bar, OBJECT(proxy),

                              &virtio_pci_config_ops,

                              proxy, "virtio-pci", size);



        pci_register_bar(&proxy->pci_dev, proxy->legacy_io_bar_idx,

                         PCI_BASE_ADDRESS_SPACE_IO, &proxy->bar);

    }

}
