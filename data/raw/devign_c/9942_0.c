void virtio_queue_notify(VirtIODevice *vdev, int n)

{

    if (n < VIRTIO_PCI_QUEUE_MAX && vdev->vq[n].vring.desc) {

        trace_virtio_queue_notify(vdev, n, &vdev->vq[n]);

        vdev->vq[n].handle_output(vdev, &vdev->vq[n]);

    }

}
