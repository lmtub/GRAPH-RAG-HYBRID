static ram_addr_t s390_virtio_device_num_vq(VirtIOS390Device *dev)

{

    VirtIODevice *vdev = dev->vdev;

    int num_vq;



    for (num_vq = 0; num_vq < VIRTIO_PCI_QUEUE_MAX; num_vq++) {

        if (!virtio_queue_get_num(vdev, num_vq)) {

            break;

        }

    }



    return num_vq;

}
