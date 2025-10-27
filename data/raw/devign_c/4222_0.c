static coroutine_fn void do_co_req(void *opaque)

{

    int ret;

    Coroutine *co;

    SheepdogReqCo *srco = opaque;

    int sockfd = srco->sockfd;

    SheepdogReq *hdr = srco->hdr;

    void *data = srco->data;

    unsigned int *wlen = srco->wlen;

    unsigned int *rlen = srco->rlen;



    co = qemu_coroutine_self();

    qemu_aio_set_fd_handler(sockfd, NULL, restart_co_req, co);



    ret = send_co_req(sockfd, hdr, data, wlen);

    if (ret < 0) {

        goto out;

    }



    qemu_aio_set_fd_handler(sockfd, restart_co_req, NULL, co);



    ret = qemu_co_recv(sockfd, hdr, sizeof(*hdr));

    if (ret < sizeof(*hdr)) {

        error_report("failed to get a rsp, %s", strerror(errno));

        ret = -errno;

        goto out;

    }



    if (*rlen > hdr->data_length) {

        *rlen = hdr->data_length;

    }



    if (*rlen) {

        ret = qemu_co_recv(sockfd, data, *rlen);

        if (ret < *rlen) {

            error_report("failed to get the data, %s", strerror(errno));

            ret = -errno;

            goto out;

        }

    }

    ret = 0;

out:

    /* there is at most one request for this sockfd, so it is safe to

     * set each handler to NULL. */

    qemu_aio_set_fd_handler(sockfd, NULL, NULL, NULL);



    srco->ret = ret;

    srco->finished = true;

}
