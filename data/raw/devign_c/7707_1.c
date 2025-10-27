void virtio_queue_set_notification(VirtQueue *vq, int enable)

{

    vq->notification = enable;

    if (vq->vdev->guest_features & (1 << VIRTIO_RING_F_EVENT_IDX)) {

        vring_avail_event(vq, vring_avail_idx(vq));

    } else if (enable) {

        vring_used_flags_unset_bit(vq, VRING_USED_F_NO_NOTIFY);

    } else {

        vring_used_flags_set_bit(vq, VRING_USED_F_NO_NOTIFY);





