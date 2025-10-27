static int virtio_scsi_vring_init(VirtIOSCSI *s, VirtQueue *vq, int n)

{

    BusState *qbus = BUS(qdev_get_parent_bus(DEVICE(s)));

    VirtioBusClass *k = VIRTIO_BUS_GET_CLASS(qbus);

    int rc;



    /* Set up virtqueue notify */

    rc = k->set_host_notifier(qbus->parent, n, true);

    if (rc != 0) {

        fprintf(stderr, "virtio-scsi: Failed to set host notifier (%d)\n",

                rc);

        s->dataplane_fenced = true;

        return rc;

    }



    virtio_queue_aio_set_host_notifier_handler(vq, s->ctx, true, true);

    return 0;

}
