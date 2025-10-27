static int nbd_receive_request(int csock, struct nbd_request *request)

{

    uint8_t buf[4 + 4 + 8 + 8 + 4];

    uint32_t magic;



    if (read_sync(csock, buf, sizeof(buf)) != sizeof(buf)) {

        LOG("read failed");

        errno = EINVAL;

        return -1;

    }



    /* Request

       [ 0 ..  3]   magic   (NBD_REQUEST_MAGIC)

       [ 4 ..  7]   type    (0 == READ, 1 == WRITE)

       [ 8 .. 15]   handle

       [16 .. 23]   from

       [24 .. 27]   len

     */



    magic = be32_to_cpup((uint32_t*)buf);

    request->type  = be32_to_cpup((uint32_t*)(buf + 4));

    request->handle = be64_to_cpup((uint64_t*)(buf + 8));

    request->from  = be64_to_cpup((uint64_t*)(buf + 16));

    request->len   = be32_to_cpup((uint32_t*)(buf + 24));



    TRACE("Got request: "

          "{ magic = 0x%x, .type = %d, from = %" PRIu64" , len = %u }",

          magic, request->type, request->from, request->len);



    if (magic != NBD_REQUEST_MAGIC) {

        LOG("invalid magic (got 0x%x)", magic);

        errno = EINVAL;

        return -1;

    }

    return 0;

}
