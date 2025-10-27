static VirtIOSCSIVring *virtio_scsi_vring_init(VirtIOSCSI *s,

                                               VirtQueue *vq,

                                               EventNotifierHandler *handler,

                                               int n)

{

    BusState *qbus = BUS(qdev_get_parent_bus(DEVICE(s)));

    VirtioBusClass *k = VIRTIO_BUS_GET_CLASS(qbus);

    VirtIOSCSIVring *r;

    int rc;



    /* Set up virtqueue notify */

    rc = k->set_host_notifier(qbus->parent, n, true);

    if (rc != 0) {

        fprintf(stderr, "virtio-scsi: Failed to set host notifier (%d)\n",

                rc);

        s->dataplane_fenced = true;

        return NULL;

    }



    r = g_new(VirtIOSCSIVring, 1);

    r->host_notifier = *virtio_queue_get_host_notifier(vq);

    r->guest_notifier = *virtio_queue_get_guest_notifier(vq);

    aio_set_event_notifier(s->ctx, &r->host_notifier, false,

                           handler);



    r->parent = s;



    if (!vring_setup(&r->vring, VIRTIO_DEVICE(s), n)) {

        fprintf(stderr, "virtio-scsi: VRing setup failed\n");

        goto fail_vring;

    }

    return r;



fail_vring:

    aio_set_event_notifier(s->ctx, &r->host_notifier, false,

                           NULL);

    k->set_host_notifier(qbus->parent, n, false);

    g_free(r);

    return NULL;

}
