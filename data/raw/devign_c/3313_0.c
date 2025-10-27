static ssize_t nc_sendv_compat(NetClientState *nc, const struct iovec *iov,

                               int iovcnt)

{

    uint8_t buffer[4096];

    size_t offset;



    offset = iov_to_buf(iov, iovcnt, 0, buffer, sizeof(buffer));



    return nc->info->receive(nc, buffer, offset);

}
