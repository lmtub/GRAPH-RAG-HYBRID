void virtio_blk_data_plane_stop(VirtIOBlockDataPlane *s)

{

    if (!s->started) {

        return;

    }

    s->started = false;

    trace_virtio_blk_data_plane_stop(s);



    /* Stop thread or cancel pending thread creation BH */

    if (s->start_bh) {

        qemu_bh_delete(s->start_bh);

        s->start_bh = NULL;

    } else {

        event_poll_notify(&s->event_poll);

        qemu_thread_join(&s->thread);

    }



    ioq_cleanup(&s->ioqueue);



    s->vdev->binding->set_host_notifier(s->vdev->binding_opaque, 0, false);



    event_poll_cleanup(&s->event_poll);



    /* Clean up guest notifier (irq) */

    s->vdev->binding->set_guest_notifiers(s->vdev->binding_opaque, 1, false);



    vring_teardown(&s->vring);

}
