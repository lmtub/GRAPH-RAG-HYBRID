static void virtio_pci_modern_regions_init(VirtIOPCIProxy *proxy)

{

    static const MemoryRegionOps common_ops = {

        .read = virtio_pci_common_read,

        .write = virtio_pci_common_write,

        .impl = {

            .min_access_size = 1,

            .max_access_size = 4,

        },

        .endianness = DEVICE_LITTLE_ENDIAN,

    };

    static const MemoryRegionOps isr_ops = {

        .read = virtio_pci_isr_read,

        .write = virtio_pci_isr_write,

        .impl = {

            .min_access_size = 1,

            .max_access_size = 4,

        },

        .endianness = DEVICE_LITTLE_ENDIAN,

    };

    static const MemoryRegionOps device_ops = {

        .read = virtio_pci_device_read,

        .write = virtio_pci_device_write,

        .impl = {

            .min_access_size = 1,

            .max_access_size = 4,

        },

        .endianness = DEVICE_LITTLE_ENDIAN,

    };

    static const MemoryRegionOps notify_ops = {

        .read = virtio_pci_notify_read,

        .write = virtio_pci_notify_write,

        .impl = {

            .min_access_size = 1,

            .max_access_size = 4,

        },

        .endianness = DEVICE_LITTLE_ENDIAN,

    };



    memory_region_init_io(&proxy->common.mr, OBJECT(proxy),

                          &common_ops,

                          proxy,

                          "virtio-pci-common", 0x1000);

    proxy->common.offset = 0x0;

    proxy->common.type = VIRTIO_PCI_CAP_COMMON_CFG;



    memory_region_init_io(&proxy->isr.mr, OBJECT(proxy),

                          &isr_ops,

                          proxy,

                          "virtio-pci-isr", 0x1000);

    proxy->isr.offset = 0x1000;

    proxy->isr.type = VIRTIO_PCI_CAP_ISR_CFG;



    memory_region_init_io(&proxy->device.mr, OBJECT(proxy),

                          &device_ops,

                          virtio_bus_get_device(&proxy->bus),

                          "virtio-pci-device", 0x1000);

    proxy->device.offset = 0x2000;

    proxy->device.type = VIRTIO_PCI_CAP_DEVICE_CFG;



    memory_region_init_io(&proxy->notify.mr, OBJECT(proxy),

                          &notify_ops,

                          virtio_bus_get_device(&proxy->bus),

                          "virtio-pci-notify",

                          QEMU_VIRTIO_PCI_QUEUE_MEM_MULT *

                          VIRTIO_QUEUE_MAX);

    proxy->notify.offset = 0x3000;

    proxy->notify.type = VIRTIO_PCI_CAP_NOTIFY_CFG;

}
