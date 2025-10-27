static int do_req(int sockfd, AioContext *aio_context, SheepdogReq *hdr,

                  void *data, unsigned int *wlen, unsigned int *rlen)

{

    Coroutine *co;

    SheepdogReqCo srco = {

        .sockfd = sockfd,

        .aio_context = aio_context,

        .hdr = hdr,

        .data = data,

        .wlen = wlen,

        .rlen = rlen,

        .ret = 0,

        .finished = false,

    };



    if (qemu_in_coroutine()) {

        do_co_req(&srco);

    } else {

        co = qemu_coroutine_create(do_co_req);

        qemu_coroutine_enter(co, &srco);

        while (!srco.finished) {

            aio_poll(aio_context, true);

        }

    }



    return srco.ret;

}
