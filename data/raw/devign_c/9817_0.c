static int coroutine_fn add_aio_request(BDRVSheepdogState *s, AIOReq *aio_req,

                           struct iovec *iov, int niov, bool create,

                           enum AIOCBState aiocb_type)

{

    int nr_copies = s->inode.nr_copies;

    SheepdogObjReq hdr;

    unsigned int wlen;

    int ret;

    uint64_t oid = aio_req->oid;

    unsigned int datalen = aio_req->data_len;

    uint64_t offset = aio_req->offset;

    uint8_t flags = aio_req->flags;

    uint64_t old_oid = aio_req->base_oid;



    if (!nr_copies) {

        error_report("bug");

    }



    memset(&hdr, 0, sizeof(hdr));



    if (aiocb_type == AIOCB_READ_UDATA) {

        wlen = 0;

        hdr.opcode = SD_OP_READ_OBJ;

        hdr.flags = flags;

    } else if (create) {

        wlen = datalen;

        hdr.opcode = SD_OP_CREATE_AND_WRITE_OBJ;

        hdr.flags = SD_FLAG_CMD_WRITE | flags;

    } else {

        wlen = datalen;

        hdr.opcode = SD_OP_WRITE_OBJ;

        hdr.flags = SD_FLAG_CMD_WRITE | flags;

    }



    if (s->cache_flags) {

        hdr.flags |= s->cache_flags;

    }



    hdr.oid = oid;

    hdr.cow_oid = old_oid;

    hdr.copies = s->inode.nr_copies;



    hdr.data_length = datalen;

    hdr.offset = offset;



    hdr.id = aio_req->id;



    qemu_co_mutex_lock(&s->lock);

    s->co_send = qemu_coroutine_self();

    qemu_aio_set_fd_handler(s->fd, co_read_response, co_write_request,

                            aio_flush_request, s);

    socket_set_cork(s->fd, 1);



    /* send a header */

    ret = qemu_co_send(s->fd, &hdr, sizeof(hdr));

    if (ret < 0) {

        qemu_co_mutex_unlock(&s->lock);

        error_report("failed to send a req, %s", strerror(errno));

        return -errno;

    }



    if (wlen) {

        ret = qemu_co_sendv(s->fd, iov, niov, aio_req->iov_offset, wlen);

        if (ret < 0) {

            qemu_co_mutex_unlock(&s->lock);

            error_report("failed to send a data, %s", strerror(errno));

            return -errno;

        }

    }



    socket_set_cork(s->fd, 0);

    qemu_aio_set_fd_handler(s->fd, co_read_response, NULL,

                            aio_flush_request, s);

    qemu_co_mutex_unlock(&s->lock);



    return 0;

}
