static coroutine_fn void nbd_read_reply_entry(void *opaque)

{

    NBDClientSession *s = opaque;

    uint64_t i;

    int ret;



    for (;;) {

        assert(s->reply.handle == 0);

        ret = nbd_receive_reply(s->ioc, &s->reply);

        if (ret < 0) {

            break;

        }



        /* There's no need for a mutex on the receive side, because the

         * handler acts as a synchronization point and ensures that only

         * one coroutine is called until the reply finishes.

         */

        i = HANDLE_TO_INDEX(s, s->reply.handle);

        if (i >= MAX_NBD_REQUESTS || !s->recv_coroutine[i]) {

            break;

        }



        /* We're woken up by the recv_coroutine itself.  Note that there

         * is no race between yielding and reentering read_reply_co.  This

         * is because:

         *

         * - if recv_coroutine[i] runs on the same AioContext, it is only

         *   entered after we yield

         *

         * - if recv_coroutine[i] runs on a different AioContext, reentering

         *   read_reply_co happens through a bottom half, which can only

         *   run after we yield.

         */

        aio_co_wake(s->recv_coroutine[i]);

        qemu_coroutine_yield();

    }

    s->read_reply_co = NULL;

}
