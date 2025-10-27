void virtio_9p_push_and_notify(V9fsPDU *pdu)

{

    V9fsState *s = pdu->s;

    V9fsVirtioState *v = container_of(s, V9fsVirtioState, state);

    VirtQueueElement *elem = &v->elems[pdu->idx];



    /* push onto queue and notify */

    virtqueue_push(v->vq, elem, pdu->size);



    /* FIXME: we should batch these completions */

    virtio_notify(VIRTIO_DEVICE(v), v->vq);

}
