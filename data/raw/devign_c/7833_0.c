void virtio_init_iov_from_pdu(V9fsPDU *pdu, struct iovec **piov,

                              unsigned int *pniov, bool is_write)

{

    V9fsState *s = pdu->s;

    V9fsVirtioState *v = container_of(s, V9fsVirtioState, state);

    VirtQueueElement *elem = &v->elems[pdu->idx];



    if (is_write) {

        *piov = elem->out_sg;

        *pniov = elem->out_num;

    } else {

        *piov = elem->in_sg;

        *pniov = elem->in_num;

    }

}
