int nbd_send_request(int csock, struct nbd_request *request)

{

    uint8_t buf[4 + 4 + 8 + 8 + 4];



    cpu_to_be32w((uint32_t*)buf, NBD_REQUEST_MAGIC);

    cpu_to_be32w((uint32_t*)(buf + 4), request->type);

    cpu_to_be64w((uint64_t*)(buf + 8), request->handle);

    cpu_to_be64w((uint64_t*)(buf + 16), request->from);

    cpu_to_be32w((uint32_t*)(buf + 24), request->len);



    TRACE("Sending request to client: "

          "{ .from = %" PRIu64", .len = %u, .handle = %" PRIu64", .type=%i}",

          request->from, request->len, request->handle, request->type);



    if (write_sync(csock, buf, sizeof(buf)) != sizeof(buf)) {

        LOG("writing to socket failed");

        errno = EINVAL;

        return -1;

    }

    return 0;

}
