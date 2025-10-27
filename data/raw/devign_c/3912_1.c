static ssize_t qio_channel_websock_decode_header(QIOChannelWebsock *ioc,

                                                 Error **errp)

{

    unsigned char opcode, fin, has_mask;

    size_t header_size;

    size_t payload_len;

    QIOChannelWebsockHeader *header =

        (QIOChannelWebsockHeader *)ioc->encinput.buffer;



    if (ioc->payload_remain) {

        error_setg(errp,

                   "Decoding header but %zu bytes of payload remain",

                   ioc->payload_remain);

        return -1;

    }

    if (ioc->encinput.offset < QIO_CHANNEL_WEBSOCK_HEADER_LEN_7_BIT) {

        /* header not complete */

        return QIO_CHANNEL_ERR_BLOCK;

    }



    fin = (header->b0 & QIO_CHANNEL_WEBSOCK_HEADER_FIELD_FIN) >>

        QIO_CHANNEL_WEBSOCK_HEADER_SHIFT_FIN;

    opcode = header->b0 & QIO_CHANNEL_WEBSOCK_HEADER_FIELD_OPCODE;

    has_mask = (header->b1 & QIO_CHANNEL_WEBSOCK_HEADER_FIELD_HAS_MASK) >>

        QIO_CHANNEL_WEBSOCK_HEADER_SHIFT_HAS_MASK;

    payload_len = header->b1 & QIO_CHANNEL_WEBSOCK_HEADER_FIELD_PAYLOAD_LEN;



    if (opcode == QIO_CHANNEL_WEBSOCK_OPCODE_CLOSE) {

        /* disconnect */

        return 0;

    }



    /* Websocket frame sanity check:

     * * Websocket fragmentation is not supported.

     * * All  websockets frames sent by a client have to be masked.

     * * Only binary encoding is supported.

     */

    if (!fin) {

        error_setg(errp, "websocket fragmentation is not supported");

        return -1;

    }

    if (!has_mask) {

        error_setg(errp, "websocket frames must be masked");

        return -1;

    }

    if (opcode != QIO_CHANNEL_WEBSOCK_OPCODE_BINARY_FRAME) {

        error_setg(errp, "only binary websocket frames are supported");

        return -1;

    }



    if (payload_len < QIO_CHANNEL_WEBSOCK_PAYLOAD_LEN_MAGIC_16_BIT) {

        ioc->payload_remain = payload_len;

        header_size = QIO_CHANNEL_WEBSOCK_HEADER_LEN_7_BIT;

        ioc->mask = header->u.m;

    } else if (payload_len == QIO_CHANNEL_WEBSOCK_PAYLOAD_LEN_MAGIC_16_BIT &&

               ioc->encinput.offset >= QIO_CHANNEL_WEBSOCK_HEADER_LEN_16_BIT) {

        ioc->payload_remain = be16_to_cpu(header->u.s16.l16);

        header_size = QIO_CHANNEL_WEBSOCK_HEADER_LEN_16_BIT;

        ioc->mask = header->u.s16.m16;

    } else if (payload_len == QIO_CHANNEL_WEBSOCK_PAYLOAD_LEN_MAGIC_64_BIT &&

               ioc->encinput.offset >= QIO_CHANNEL_WEBSOCK_HEADER_LEN_64_BIT) {

        ioc->payload_remain = be64_to_cpu(header->u.s64.l64);

        header_size = QIO_CHANNEL_WEBSOCK_HEADER_LEN_64_BIT;

        ioc->mask = header->u.s64.m64;

    } else {

        /* header not complete */

        return QIO_CHANNEL_ERR_BLOCK;

    }



    buffer_advance(&ioc->encinput, header_size);

    return 1;

}
