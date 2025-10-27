static void vhost_virtqueue_stop(struct vhost_dev *dev,

                                    struct VirtIODevice *vdev,

                                    struct vhost_virtqueue *vq,

                                    unsigned idx)

{

    int vhost_vq_index = dev->vhost_ops->vhost_get_vq_index(dev, idx);

    struct vhost_vring_state state = {

        .index = vhost_vq_index,

    };

    int r;



    r = dev->vhost_ops->vhost_get_vring_base(dev, &state);

    if (r < 0) {

        VHOST_OPS_DEBUG("vhost VQ %d ring restore failed: %d", idx, r);





    } else {

        virtio_queue_set_last_avail_idx(vdev, idx, state.num);

    }

    virtio_queue_invalidate_signalled_used(vdev, idx);

    virtio_queue_update_used_idx(vdev, idx);



    /* In the cross-endian case, we need to reset the vring endianness to

     * native as legacy devices expect so by default.


    if (vhost_needs_vring_endian(vdev)) {

        vhost_virtqueue_set_vring_endian_legacy(dev,

                                                !virtio_is_big_endian(vdev),

                                                vhost_vq_index);

    }



    vhost_memory_unmap(dev, vq->used, virtio_queue_get_used_size(vdev, idx),

                       1, virtio_queue_get_used_size(vdev, idx));

    vhost_memory_unmap(dev, vq->avail, virtio_queue_get_avail_size(vdev, idx),

                       0, virtio_queue_get_avail_size(vdev, idx));

    vhost_memory_unmap(dev, vq->desc, virtio_queue_get_desc_size(vdev, idx),

                       0, virtio_queue_get_desc_size(vdev, idx));

}