static ssize_t nbd_receive_request(QIOChannel *ioc, NBDRequest *request)

{

    uint8_t buf[NBD_REQUEST_SIZE];

    uint32_t magic;

    ssize_t ret;



    ret = read_sync(ioc, buf, sizeof(buf), NULL);

    if (ret < 0) {

        return ret;

    }



    /* Request

       [ 0 ..  3]   magic   (NBD_REQUEST_MAGIC)

       [ 4 ..  5]   flags   (NBD_CMD_FLAG_FUA, ...)

       [ 6 ..  7]   type    (NBD_CMD_READ, ...)

       [ 8 .. 15]   handle

       [16 .. 23]   from

       [24 .. 27]   len

     */



    magic = ldl_be_p(buf);

    request->flags  = lduw_be_p(buf + 4);

    request->type   = lduw_be_p(buf + 6);

    request->handle = ldq_be_p(buf + 8);

    request->from   = ldq_be_p(buf + 16);

    request->len    = ldl_be_p(buf + 24);



    TRACE("Got request: { magic = 0x%" PRIx32 ", .flags = %" PRIx16

          ", .type = %" PRIx16 ", from = %" PRIu64 ", len = %" PRIu32 " }",

          magic, request->flags, request->type, request->from, request->len);



    if (magic != NBD_REQUEST_MAGIC) {

        LOG("invalid magic (got 0x%" PRIx32 ")", magic);

        return -EINVAL;

    }

    return 0;

}
