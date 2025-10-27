void virtio_queue_notify(VirtIODevice *vdev, int n)

{

    if (n < VIRTIO_PCI_QUEUE_MAX) {

        virtio_queue_notify_vq(&vdev->vq[n]);

    }

}
