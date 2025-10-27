ssize_t nbd_receive_reply(QIOChannel *ioc, NBDReply *reply, Error **errp)

{

    uint8_t buf[NBD_REPLY_SIZE];

    uint32_t magic;

    ssize_t ret;



    ret = read_sync_eof(ioc, buf, sizeof(buf), errp);

    if (ret <= 0) {

        return ret;

    }



    if (ret != sizeof(buf)) {

        error_setg(errp, "read failed");

        return -EINVAL;

    }



    /* Reply

       [ 0 ..  3]    magic   (NBD_REPLY_MAGIC)

       [ 4 ..  7]    error   (0 == no error)

       [ 7 .. 15]    handle

     */



    magic = ldl_be_p(buf);

    reply->error  = ldl_be_p(buf + 4);

    reply->handle = ldq_be_p(buf + 8);



    reply->error = nbd_errno_to_system_errno(reply->error);



    if (reply->error == ESHUTDOWN) {

        /* This works even on mingw which lacks a native ESHUTDOWN */

        error_setg(errp, "server shutting down");

        return -EINVAL;

    }

    TRACE("Got reply: { magic = 0x%" PRIx32 ", .error = % " PRId32

          ", handle = %" PRIu64" }",

          magic, reply->error, reply->handle);



    if (magic != NBD_REPLY_MAGIC) {

        error_setg(errp, "invalid magic (got 0x%" PRIx32 ")", magic);

        return -EINVAL;

    }

    return sizeof(buf);

}
