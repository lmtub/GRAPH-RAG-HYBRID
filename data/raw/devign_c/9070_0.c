void virtio_scsi_dataplane_stop(VirtIOSCSI *s)

{

    BusState *qbus = BUS(qdev_get_parent_bus(DEVICE(s)));

    VirtioBusClass *k = VIRTIO_BUS_GET_CLASS(qbus);

    VirtIOSCSICommon *vs = VIRTIO_SCSI_COMMON(s);

    int i;



    /* Better luck next time. */

    if (s->dataplane_fenced) {

        s->dataplane_fenced = false;

        return;

    }

    if (!s->dataplane_started || s->dataplane_stopping) {

        return;

    }

    s->dataplane_stopping = true;

    assert(s->ctx == iothread_get_aio_context(vs->conf.iothread));



    aio_context_acquire(s->ctx);



    aio_set_event_notifier(s->ctx, &s->ctrl_vring->host_notifier,

                           false, NULL);

    aio_set_event_notifier(s->ctx, &s->event_vring->host_notifier,

                           false, NULL);

    for (i = 0; i < vs->conf.num_queues; i++) {

        aio_set_event_notifier(s->ctx, &s->cmd_vrings[i]->host_notifier,

                               false, NULL);

    }



    blk_drain_all(); /* ensure there are no in-flight requests */



    aio_context_release(s->ctx);



    /* Sync vring state back to virtqueue so that non-dataplane request

     * processing can continue when we disable the host notifier below.

     */

    virtio_scsi_vring_teardown(s);



    for (i = 0; i < vs->conf.num_queues + 2; i++) {

        k->set_host_notifier(qbus->parent, i, false);

    }



    /* Clean up guest notifier (irq) */

    k->set_guest_notifiers(qbus->parent, vs->conf.num_queues + 2, false);

    s->dataplane_stopping = false;

    s->dataplane_started = false;

}
