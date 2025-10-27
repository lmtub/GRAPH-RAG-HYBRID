static void qio_channel_websock_write_close(QIOChannelWebsock *ioc,

                                            uint16_t code, const char *reason)

{

    struct iovec iov;

    buffer_reserve(&ioc->rawoutput, 2 + (reason ? strlen(reason) : 0));

    *(uint16_t *)(ioc->rawoutput.buffer + ioc->rawoutput.offset) =

        cpu_to_be16(code);

    ioc->rawoutput.offset += 2;

    if (reason) {

        buffer_append(&ioc->rawoutput, reason, strlen(reason));

    }

    iov.iov_base = ioc->rawoutput.buffer;

    iov.iov_len = ioc->rawoutput.offset;

    qio_channel_websock_encode(ioc, QIO_CHANNEL_WEBSOCK_OPCODE_CLOSE,

                               &iov, 1, iov.iov_len);

    buffer_reset(&ioc->rawoutput);

    qio_channel_websock_write_wire(ioc, NULL);

    qio_channel_shutdown(ioc->master, QIO_CHANNEL_SHUTDOWN_BOTH, NULL);

}
