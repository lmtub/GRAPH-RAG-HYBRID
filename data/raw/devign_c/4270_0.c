static void *rndis_queue_response(USBNetState *s, unsigned int length)

{

    struct rndis_response *r =

            qemu_mallocz(sizeof(struct rndis_response) + length);



    TAILQ_INSERT_TAIL(&s->rndis_resp, r, entries);

    r->length = length;



    return &r->buf[0];

}
