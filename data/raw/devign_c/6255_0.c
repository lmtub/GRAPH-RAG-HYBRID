void virtio_input_send(VirtIOInput *vinput, virtio_input_event *event)

{

    VirtQueueElement *elem;

    unsigned have, need;

    int i, len;



    if (!vinput->active) {

        return;

    }



    /* queue up events ... */

    if (vinput->qindex == vinput->qsize) {

        vinput->qsize++;

        vinput->queue = g_realloc(vinput->queue, vinput->qsize *

                                  sizeof(virtio_input_event));

    }

    vinput->queue[vinput->qindex++] = *event;



    /* ... until we see a report sync ... */

    if (event->type != cpu_to_le16(EV_SYN) ||

        event->code != cpu_to_le16(SYN_REPORT)) {

        return;

    }



    /* ... then check available space ... */

    need = sizeof(virtio_input_event) * vinput->qindex;

    virtqueue_get_avail_bytes(vinput->evt, &have, NULL, need, 0);

    if (have < need) {

        vinput->qindex = 0;

        trace_virtio_input_queue_full();

        return;

    }



    /* ... and finally pass them to the guest */

    for (i = 0; i < vinput->qindex; i++) {

        elem = virtqueue_pop(vinput->evt, sizeof(VirtQueueElement));

        if (!elem) {

            /* should not happen, we've checked for space beforehand */

            fprintf(stderr, "%s: Huh?  No vq elem available ...\n", __func__);

            return;

        }

        len = iov_from_buf(elem->in_sg, elem->in_num,

                           0, vinput->queue+i, sizeof(virtio_input_event));

        virtqueue_push(vinput->evt, elem, len);

        g_free(elem);

    }

    virtio_notify(VIRTIO_DEVICE(vinput), vinput->evt);

    vinput->qindex = 0;

}
