static void balloon_stats_poll_cb(void *opaque)

{

    VirtIOBalloon *s = opaque;

    VirtIODevice *vdev = VIRTIO_DEVICE(s);



    if (!balloon_stats_supported(s)) {

        /* re-schedule */

        balloon_stats_change_timer(s, s->stats_poll_interval);

        return;

    }



    virtqueue_push(s->svq, s->stats_vq_elem, s->stats_vq_offset);

    virtio_notify(vdev, s->svq);

    g_free(s->stats_vq_elem);

    s->stats_vq_elem = NULL;

}
