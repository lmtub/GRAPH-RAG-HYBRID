static void nbd_reply_ready(void *opaque)

{

    BDRVNBDState *s = opaque;

    uint64_t i;



    if (s->reply.handle == 0) {

        /* No reply already in flight.  Fetch a header.  */

        if (nbd_receive_reply(s->sock, &s->reply) < 0) {

            s->reply.handle = 0;

            goto fail;

        }

    }



    /* There's no need for a mutex on the receive side, because the

     * handler acts as a synchronization point and ensures that only

     * one coroutine is called until the reply finishes.  */

    i = HANDLE_TO_INDEX(s, s->reply.handle);

    if (i >= MAX_NBD_REQUESTS) {

        goto fail;

    }



    if (s->recv_coroutine[i]) {

        qemu_coroutine_enter(s->recv_coroutine[i], NULL);

        return;

    }



fail:

    for (i = 0; i < MAX_NBD_REQUESTS; i++) {

        if (s->recv_coroutine[i]) {

            qemu_coroutine_enter(s->recv_coroutine[i], NULL);

        }

    }

}
