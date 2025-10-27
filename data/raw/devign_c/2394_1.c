static void virtio_pci_config_write(void *opaque, hwaddr addr,

                                    uint64_t val, unsigned size)

{

    VirtIOPCIProxy *proxy = opaque;

    uint32_t config = VIRTIO_PCI_CONFIG(&proxy->pci_dev);

    VirtIODevice *vdev = virtio_bus_get_device(&proxy->bus);

    if (addr < config) {

        virtio_ioport_write(proxy, addr, val);

        return;

    }

    addr -= config;

    /*

     * Virtio-PCI is odd. Ioports are LE but config space is target native

     * endian.

     */

    switch (size) {

    case 1:

        virtio_config_writeb(vdev, addr, val);

        break;

    case 2:

        if (virtio_is_big_endian()) {

            val = bswap16(val);

        }

        virtio_config_writew(vdev, addr, val);

        break;

    case 4:

        if (virtio_is_big_endian()) {

            val = bswap32(val);

        }

        virtio_config_writel(vdev, addr, val);

        break;

    }

}
