static coroutine_fn int send_co_req(int sockfd, SheepdogReq *hdr, void *data,

                                    unsigned int *wlen)

{

    int ret;



    ret = qemu_co_send(sockfd, hdr, sizeof(*hdr));

    if (ret < sizeof(*hdr)) {

        error_report("failed to send a req, %s", strerror(errno));

        return ret;

    }



    ret = qemu_co_send(sockfd, data, *wlen);

    if (ret < *wlen) {

        error_report("failed to send a req, %s", strerror(errno));

    }



    return ret;

}
