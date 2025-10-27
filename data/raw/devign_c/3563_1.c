static void co_read_response(void *opaque)

{

    BDRVSheepdogState *s = opaque;



    if (!s->co_recv) {

        s->co_recv = qemu_coroutine_create(aio_read_response);

    }



    qemu_coroutine_enter(s->co_recv, opaque);

}
