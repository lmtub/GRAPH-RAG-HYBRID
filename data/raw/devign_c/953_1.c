static void notify_guest_bh(void *opaque)

{

    VirtIOBlockDataPlane *s = opaque;

    unsigned nvqs = s->conf->num_queues;

    unsigned long bitmap[BITS_TO_LONGS(nvqs)];

    unsigned j;



    memcpy(bitmap, s->batch_notify_vqs, sizeof(bitmap));

    memset(s->batch_notify_vqs, 0, sizeof(bitmap));



    for (j = 0; j < nvqs; j += BITS_PER_LONG) {

        unsigned long bits = bitmap[j];



        while (bits != 0) {

            unsigned i = j + ctzl(bits);

            VirtQueue *vq = virtio_get_queue(s->vdev, i);



            if (virtio_should_notify(s->vdev, vq)) {

                event_notifier_set(virtio_queue_get_guest_notifier(vq));

            }



            bits &= bits - 1; /* clear right-most bit */

        }

    }

}
