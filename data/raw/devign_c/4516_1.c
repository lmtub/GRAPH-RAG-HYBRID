void virtio_scsi_dataplane_start(VirtIOSCSI *s)

{

    int i;

    int rc;

    BusState *qbus = BUS(qdev_get_parent_bus(DEVICE(s)));

    VirtioBusClass *k = VIRTIO_BUS_GET_CLASS(qbus);

    VirtIOSCSICommon *vs = VIRTIO_SCSI_COMMON(s);



    if (s->dataplane_started ||

        s->dataplane_starting ||

        s->dataplane_fenced ||

        s->ctx != iothread_get_aio_context(vs->conf.iothread)) {

        return;

    }



    s->dataplane_starting = true;



    /* Set up guest notifier (irq) */

    rc = k->set_guest_notifiers(qbus->parent, vs->conf.num_queues + 2, true);

    if (rc != 0) {

        fprintf(stderr, "virtio-scsi: Failed to set guest notifiers (%d), "

                "ensure -enable-kvm is set\n", rc);

        goto fail_guest_notifiers;

    }



    aio_context_acquire(s->ctx);

    rc = virtio_scsi_vring_init(s, vs->ctrl_vq, 0);

    if (rc) {

        goto fail_vrings;

    }

    rc = virtio_scsi_vring_init(s, vs->event_vq, 1);

    if (rc) {

        goto fail_vrings;

    }

    for (i = 0; i < vs->conf.num_queues; i++) {

        rc = virtio_scsi_vring_init(s, vs->cmd_vqs[i], i + 2);

        if (rc) {

            goto fail_vrings;

        }

    }



    s->dataplane_starting = false;

    s->dataplane_started = true;

    aio_context_release(s->ctx);

    return;



fail_vrings:

    virtio_scsi_clear_aio(s);

    aio_context_release(s->ctx);

    for (i = 0; i < vs->conf.num_queues + 2; i++) {

        k->set_host_notifier(qbus->parent, i, false);

    }

    k->set_guest_notifiers(qbus->parent, vs->conf.num_queues + 2, false);

fail_guest_notifiers:

    s->dataplane_fenced = true;

    s->dataplane_starting = false;

    s->dataplane_started = true;

}
