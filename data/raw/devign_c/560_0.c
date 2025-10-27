void virtio_queue_set_num(VirtIODevice *vdev, int n, int num)

{

    if (num <= VIRTQUEUE_MAX_SIZE) {

        vdev->vq[n].vring.num = num;

        virtqueue_init(&vdev->vq[n]);

    }

}
