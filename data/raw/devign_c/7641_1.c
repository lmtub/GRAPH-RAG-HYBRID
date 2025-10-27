static void nbd_coroutine_end(NbdClientSession *s,

    struct nbd_request *request)

{

    int i = HANDLE_TO_INDEX(s, request->handle);

    s->recv_coroutine[i] = NULL;

    if (s->in_flight-- == MAX_NBD_REQUESTS) {

        qemu_co_mutex_unlock(&s->free_sema);

    }

}
