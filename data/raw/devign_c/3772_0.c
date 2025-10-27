static ssize_t nbd_receive_request(QIOChannel *ioc, struct nbd_request *request)

{

    uint8_t buf[NBD_REQUEST_SIZE];

    uint32_t magic;

    ssize_t ret;



    ret = read_sync(ioc, buf, sizeof(buf));

    if (ret < 0) {

        return ret;

    }



    if (ret != sizeof(buf)) {

        LOG("read failed");

        return -EINVAL;

    }



    /* Request

       [ 0 ..  3]   magic   (NBD_REQUEST_MAGIC)

       [ 4 ..  7]   type    (0 == READ, 1 == WRITE)

       [ 8 .. 15]   handle

       [16 .. 23]   from

       [24 .. 27]   len

     */



    magic = ldl_be_p(buf);

    request->type   = ldl_be_p(buf + 4);

    request->handle = ldq_be_p(buf + 8);

    request->from   = ldq_be_p(buf + 16);

    request->len    = ldl_be_p(buf + 24);



    TRACE("Got request: { magic = 0x%" PRIx32 ", .type = %" PRIx32

          ", from = %" PRIu64 " , len = %" PRIu32 " }",

          magic, request->type, request->from, request->len);



    if (magic != NBD_REQUEST_MAGIC) {

        LOG("invalid magic (got 0x%" PRIx32 ")", magic);

        return -EINVAL;

    }

    return 0;

}
