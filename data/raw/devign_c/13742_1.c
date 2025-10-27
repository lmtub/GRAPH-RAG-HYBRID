void virtio_reset(void *opaque)

{

    VirtIODevice *vdev = opaque;

    VirtioDeviceClass *k = VIRTIO_DEVICE_GET_CLASS(vdev);

    int i;



    virtio_set_status(vdev, 0);

    if (current_cpu) {

        /* Guest initiated reset */

        vdev->device_endian = virtio_current_cpu_endian();

    } else {

        /* System reset */

        vdev->device_endian = virtio_default_endian();

    }



    if (k->reset) {

        k->reset(vdev);

    }




    vdev->guest_features = 0;

    vdev->queue_sel = 0;

    vdev->status = 0;

    vdev->isr = 0;

    vdev->config_vector = VIRTIO_NO_VECTOR;

    virtio_notify_vector(vdev, vdev->config_vector);



    for(i = 0; i < VIRTIO_QUEUE_MAX; i++) {

        vdev->vq[i].vring.desc = 0;

        vdev->vq[i].vring.avail = 0;

        vdev->vq[i].vring.used = 0;

        vdev->vq[i].last_avail_idx = 0;

        vdev->vq[i].shadow_avail_idx = 0;

        vdev->vq[i].used_idx = 0;

        virtio_queue_set_vector(vdev, i, VIRTIO_NO_VECTOR);

        vdev->vq[i].signalled_used = 0;

        vdev->vq[i].signalled_used_valid = false;

        vdev->vq[i].notification = true;

        vdev->vq[i].vring.num = vdev->vq[i].vring.num_default;

        vdev->vq[i].inuse = 0;

    }

}