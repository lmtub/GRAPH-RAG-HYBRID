bool vring_should_notify(VirtIODevice *vdev, Vring *vring)

{

    uint16_t old, new;

    bool v;

    /* Flush out used index updates. This is paired

     * with the barrier that the Guest executes when enabling

     * interrupts. */

    smp_mb();



    if ((vdev->guest_features & (1 << VIRTIO_F_NOTIFY_ON_EMPTY)) &&

        unlikely(!vring_more_avail(vdev, vring))) {

        return true;

    }



    if (!(vdev->guest_features & (1 << VIRTIO_RING_F_EVENT_IDX))) {

        return !(vring_get_avail_flags(vdev, vring) &

                 VRING_AVAIL_F_NO_INTERRUPT);

    }

    old = vring->signalled_used;

    v = vring->signalled_used_valid;

    new = vring->signalled_used = vring->last_used_idx;

    vring->signalled_used_valid = true;



    if (unlikely(!v)) {

        return true;

    }



    return vring_need_event(vring_used_event(&vring->vr), new, old);

}
