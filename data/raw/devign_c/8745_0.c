static void virtio_pci_vector_mask(PCIDevice *dev, unsigned vector)

{

    VirtIOPCIProxy *proxy = container_of(dev, VirtIOPCIProxy, pci_dev);

    VirtIODevice *vdev = virtio_bus_get_device(&proxy->bus);

    VirtQueue *vq = virtio_vector_first_queue(vdev, vector);

    int index;



    while (vq) {

        index = virtio_get_queue_index(vq);

        if (!virtio_queue_get_num(vdev, index)) {

            break;

        }

        virtio_pci_vq_vector_mask(proxy, index, vector);

        vq = virtio_vector_next_queue(vq);

    }

}
