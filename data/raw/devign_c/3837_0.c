ssize_t nbd_send_request(QIOChannel *ioc, NBDRequest *request)

{

    uint8_t buf[NBD_REQUEST_SIZE];



    TRACE("Sending request to server: "

          "{ .from = %" PRIu64", .len = %" PRIu32 ", .handle = %" PRIu64

          ", .flags = %" PRIx16 ", .type = %" PRIu16 " }",

          request->from, request->len, request->handle,

          request->flags, request->type);



    stl_be_p(buf, NBD_REQUEST_MAGIC);

    stw_be_p(buf + 4, request->flags);

    stw_be_p(buf + 6, request->type);

    stq_be_p(buf + 8, request->handle);

    stq_be_p(buf + 16, request->from);

    stl_be_p(buf + 24, request->len);



    return write_sync(ioc, buf, sizeof(buf), NULL);

}
