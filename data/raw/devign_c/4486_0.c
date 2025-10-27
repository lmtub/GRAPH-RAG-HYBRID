static int virtio_pci_ioeventfd_assign(DeviceState *d, EventNotifier *notifier,

                                       int n, bool assign)

{

    VirtIOPCIProxy *proxy = to_virtio_pci_proxy(d);

    VirtIODevice *vdev = virtio_bus_get_device(&proxy->bus);

    VirtQueue *vq = virtio_get_queue(vdev, n);

    bool legacy = !(proxy->flags & VIRTIO_PCI_FLAG_DISABLE_LEGACY);

    bool modern = !(proxy->flags & VIRTIO_PCI_FLAG_DISABLE_MODERN);

    bool fast_mmio = kvm_ioeventfd_any_length_enabled();

    bool modern_pio = proxy->flags & VIRTIO_PCI_FLAG_MODERN_PIO_NOTIFY;

    MemoryRegion *modern_mr = &proxy->notify.mr;

    MemoryRegion *modern_notify_mr = &proxy->notify_pio.mr;

    MemoryRegion *legacy_mr = &proxy->bar;

    hwaddr modern_addr = QEMU_VIRTIO_PCI_QUEUE_MEM_MULT *

                         virtio_get_queue_index(vq);

    hwaddr legacy_addr = VIRTIO_PCI_QUEUE_NOTIFY;



    if (assign) {

        if (modern) {

            if (fast_mmio) {

                memory_region_add_eventfd(modern_mr, modern_addr, 0,

                                          false, n, notifier);

            } else {

                memory_region_add_eventfd(modern_mr, modern_addr, 2,

                                          false, n, notifier);

            }

            if (modern_pio) {

                memory_region_add_eventfd(modern_notify_mr, 0, 2,

                                              true, n, notifier);

            }

        }

        if (legacy) {

            memory_region_add_eventfd(legacy_mr, legacy_addr, 2,

                                      true, n, notifier);

        }

    } else {

        if (modern) {

            if (fast_mmio) {

                memory_region_del_eventfd(modern_mr, modern_addr, 0,

                                          false, n, notifier);

            } else {

                memory_region_del_eventfd(modern_mr, modern_addr, 2,

                                          false, n, notifier);

            }

            if (modern_pio) {

                memory_region_del_eventfd(modern_notify_mr, 0, 2,

                                          true, n, notifier);

            }

        }

        if (legacy) {

            memory_region_del_eventfd(legacy_mr, legacy_addr, 2,

                                      true, n, notifier);

        }

    }

    return 0;

}
