int nbd_receive_reply(int csock, struct nbd_reply *reply)

{

    uint8_t buf[NBD_REPLY_SIZE];

    uint32_t magic;



    memset(buf, 0xAA, sizeof(buf));



    if (read_sync(csock, buf, sizeof(buf)) != sizeof(buf)) {

        LOG("read failed");

        errno = EINVAL;

        return -1;

    }



    /* Reply

       [ 0 ..  3]    magic   (NBD_REPLY_MAGIC)

       [ 4 ..  7]    error   (0 == no error)

       [ 7 .. 15]    handle

     */



    magic = be32_to_cpup((uint32_t*)buf);

    reply->error  = be32_to_cpup((uint32_t*)(buf + 4));

    reply->handle = be64_to_cpup((uint64_t*)(buf + 8));



    TRACE("Got reply: "

          "{ magic = 0x%x, .error = %d, handle = %" PRIu64" }",

          magic, reply->error, reply->handle);



    if (magic != NBD_REPLY_MAGIC) {

        LOG("invalid magic (got 0x%x)", magic);

        errno = EINVAL;

        return -1;

    }

    return 0;

}
