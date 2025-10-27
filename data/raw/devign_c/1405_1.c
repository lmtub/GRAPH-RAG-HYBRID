static void qio_channel_websock_encode(QIOChannelWebsock *ioc)

{

    size_t header_size;

    union {

        char buf[QIO_CHANNEL_WEBSOCK_HEADER_LEN_64_BIT];

        QIOChannelWebsockHeader ws;

    } header;



    if (!ioc->rawoutput.offset) {

        return;

    }



    header.ws.b0 = (1 << QIO_CHANNEL_WEBSOCK_HEADER_SHIFT_FIN) |

        (QIO_CHANNEL_WEBSOCK_OPCODE_BINARY_FRAME &

         QIO_CHANNEL_WEBSOCK_HEADER_FIELD_OPCODE);

    if (ioc->rawoutput.offset <

        QIO_CHANNEL_WEBSOCK_PAYLOAD_LEN_THRESHOLD_7_BIT) {

        header.ws.b1 = (uint8_t)ioc->rawoutput.offset;

        header_size = QIO_CHANNEL_WEBSOCK_HEADER_LEN_7_BIT;

    } else if (ioc->rawoutput.offset <

               QIO_CHANNEL_WEBSOCK_PAYLOAD_LEN_THRESHOLD_16_BIT) {

        header.ws.b1 = QIO_CHANNEL_WEBSOCK_PAYLOAD_LEN_MAGIC_16_BIT;

        header.ws.u.s16.l16 = cpu_to_be16((uint16_t)ioc->rawoutput.offset);

        header_size = QIO_CHANNEL_WEBSOCK_HEADER_LEN_16_BIT;

    } else {

        header.ws.b1 = QIO_CHANNEL_WEBSOCK_PAYLOAD_LEN_MAGIC_64_BIT;

        header.ws.u.s64.l64 = cpu_to_be64(ioc->rawoutput.offset);

        header_size = QIO_CHANNEL_WEBSOCK_HEADER_LEN_64_BIT;

    }

    header_size -= QIO_CHANNEL_WEBSOCK_HEADER_LEN_MASK;



    buffer_reserve(&ioc->encoutput, header_size + ioc->rawoutput.offset);

    buffer_append(&ioc->encoutput, header.buf, header_size);

    buffer_append(&ioc->encoutput, ioc->rawoutput.buffer,

                  ioc->rawoutput.offset);

    buffer_reset(&ioc->rawoutput);

}
