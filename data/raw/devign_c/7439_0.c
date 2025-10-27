void virtio_blk_data_plane_stop(VirtIOBlockDataPlane *s)

{

    BusState *qbus = BUS(qdev_get_parent_bus(DEVICE(s->vdev)));

    VirtioBusClass *k = VIRTIO_BUS_GET_CLASS(qbus);

    VirtIOBlock *vblk = VIRTIO_BLK(s->vdev);

    unsigned i;

    unsigned nvqs = s->conf->num_queues;



    if (!vblk->dataplane_started || s->stopping) {

        return;

    }



    /* Better luck next time. */

    if (vblk->dataplane_disabled) {

        vblk->dataplane_disabled = false;

        vblk->dataplane_started = false;

        return;

    }

    s->stopping = true;

    trace_virtio_blk_data_plane_stop(s);



    aio_context_acquire(s->ctx);



    /* Stop notifications for new requests from guest */

    for (i = 0; i < nvqs; i++) {

        VirtQueue *vq = virtio_get_queue(s->vdev, i);



        virtio_queue_aio_set_host_notifier_handler(vq, s->ctx, NULL);

    }



    /* Drain and switch bs back to the QEMU main loop */

    blk_set_aio_context(s->conf->conf.blk, qemu_get_aio_context());



    aio_context_release(s->ctx);



    for (i = 0; i < nvqs; i++) {

        virtio_bus_set_host_notifier(VIRTIO_BUS(qbus), i, false);

    }



    /* Clean up guest notifier (irq) */

    k->set_guest_notifiers(qbus->parent, nvqs, false);



    vblk->dataplane_started = false;

    s->stopping = false;

}
