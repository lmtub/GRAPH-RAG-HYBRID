static coroutine_fn void reconnect_to_sdog(void *opaque)

{

    Error *local_err = NULL;

    BDRVSheepdogState *s = opaque;

    AIOReq *aio_req, *next;



    aio_set_fd_handler(s->aio_context, s->fd, NULL, NULL, NULL);

    close(s->fd);

    s->fd = -1;



    /* Wait for outstanding write requests to be completed. */

    while (s->co_send != NULL) {

        co_write_request(opaque);

    }



    /* Try to reconnect the sheepdog server every one second. */

    while (s->fd < 0) {

        s->fd = get_sheep_fd(s, &local_err);

        if (s->fd < 0) {

            DPRINTF("Wait for connection to be established\n");

            error_report("%s", error_get_pretty(local_err));

            error_free(local_err);

            co_aio_sleep_ns(bdrv_get_aio_context(s->bs), QEMU_CLOCK_REALTIME,

                            1000000000ULL);

        }

    };



    /*

     * Now we have to resend all the request in the inflight queue.  However,

     * resend_aioreq() can yield and newly created requests can be added to the

     * inflight queue before the coroutine is resumed.  To avoid mixing them, we

     * have to move all the inflight requests to the failed queue before

     * resend_aioreq() is called.

     */

    QLIST_FOREACH_SAFE(aio_req, &s->inflight_aio_head, aio_siblings, next) {

        QLIST_REMOVE(aio_req, aio_siblings);

        QLIST_INSERT_HEAD(&s->failed_aio_head, aio_req, aio_siblings);

    }



    /* Resend all the failed aio requests. */

    while (!QLIST_EMPTY(&s->failed_aio_head)) {

        aio_req = QLIST_FIRST(&s->failed_aio_head);

        QLIST_REMOVE(aio_req, aio_siblings);

        QLIST_INSERT_HEAD(&s->inflight_aio_head, aio_req, aio_siblings);

        resend_aioreq(s, aio_req);

    }

}
