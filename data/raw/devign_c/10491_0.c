static int virtio_pci_vector_unmask(PCIDevice *dev, unsigned vector,

                                    MSIMessage msg)

{

    VirtIOPCIProxy *proxy = container_of(dev, VirtIOPCIProxy, pci_dev);

    VirtIODevice *vdev = virtio_bus_get_device(&proxy->bus);

    VirtQueue *vq = virtio_vector_first_queue(vdev, vector);

    int ret, index, unmasked = 0;



    while (vq) {

        index = virtio_get_queue_index(vq);

        if (!virtio_queue_get_num(vdev, index)) {

            break;

        }

        ret = virtio_pci_vq_vector_unmask(proxy, index, vector, msg);

        if (ret < 0) {

            goto undo;

        }

        vq = virtio_vector_next_queue(vq);

        ++unmasked;

    }



    return 0;



undo:

    vq = virtio_vector_first_queue(vdev, vector);

    while (vq && --unmasked >= 0) {

        index = virtio_get_queue_index(vq);

        virtio_pci_vq_vector_mask(proxy, index, vector);

        vq = virtio_vector_next_queue(vq);

    }

    return ret;

}
