bool vring_enable_notification(VirtIODevice *vdev, Vring *vring)

{

    if (vdev->guest_features & (1 << VIRTIO_RING_F_EVENT_IDX)) {

        vring_avail_event(&vring->vr) = vring->vr.avail->idx;

    } else {

        vring_clear_used_flags(vdev, vring, VRING_USED_F_NO_NOTIFY);

    }

    smp_mb(); /* ensure update is seen before reading avail_idx */

    return !vring_more_avail(vdev, vring);

}
