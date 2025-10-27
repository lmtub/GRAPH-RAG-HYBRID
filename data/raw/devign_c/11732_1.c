static void nbd_coroutine_start(NbdClientSession *s,

   struct nbd_request *request)

{

    /* Poor man semaphore.  The free_sema is locked when no other request

     * can be accepted, and unlocked after receiving one reply.  */

    if (s->in_flight >= MAX_NBD_REQUESTS - 1) {

        qemu_co_mutex_lock(&s->free_sema);

        assert(s->in_flight < MAX_NBD_REQUESTS);

    }

    s->in_flight++;



    /* s->recv_coroutine[i] is set as soon as we get the send_lock.  */

}
