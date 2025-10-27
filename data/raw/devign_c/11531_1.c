static ssize_t nc_sendv_compat(NetClientState *nc, const struct iovec *iov,

                               int iovcnt, unsigned flags)

{

    uint8_t *buf = NULL;

    uint8_t *buffer;

    size_t offset;

    ssize_t ret;



    if (iovcnt == 1) {

        buffer = iov[0].iov_base;

        offset = iov[0].iov_len;

    } else {

        buf = g_new(uint8_t, NET_BUFSIZE);

        buffer = buf;

        offset = iov_to_buf(iov, iovcnt, 0, buf, NET_BUFSIZE);

    }



    if (flags & QEMU_NET_PACKET_FLAG_RAW && nc->info->receive_raw) {

        ret = nc->info->receive_raw(nc, buffer, offset);

    } else {

        ret = nc->info->receive(nc, buffer, offset);

    }



    g_free(buf);

    return ret;

}
