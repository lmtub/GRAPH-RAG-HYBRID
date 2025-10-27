static VirtIOSCSIVring *virtio_scsi_vring_init(VirtIOSCSI *s,

                                               VirtQueue *vq,

                                               EventNotifierHandler *handler,

                                               int n)

{

    BusState *qbus = BUS(qdev_get_parent_bus(DEVICE(s)));

    VirtioBusClass *k = VIRTIO_BUS_GET_CLASS(qbus);

    VirtIOSCSIVring *r = g_slice_new(VirtIOSCSIVring);

    int rc;



    /* Set up virtqueue notify */

    rc = k->set_host_notifier(qbus->parent, n, true);

    if (rc != 0) {

        fprintf(stderr, "virtio-scsi: Failed to set host notifier (%d)\n",

                rc);

        exit(1);

    }

    r->host_notifier = *virtio_queue_get_host_notifier(vq);

    r->guest_notifier = *virtio_queue_get_guest_notifier(vq);

    aio_set_event_notifier(s->ctx, &r->host_notifier, handler);



    r->parent = s;



    if (!vring_setup(&r->vring, VIRTIO_DEVICE(s), n)) {

        fprintf(stderr, "virtio-scsi: VRing setup failed\n");

        exit(1);

    }

    return r;

}
