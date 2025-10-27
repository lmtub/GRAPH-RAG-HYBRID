static ssize_t qio_channel_websock_read_wire(QIOChannelWebsock *ioc,

                                             Error **errp)

{

    ssize_t ret;



    if (ioc->encinput.offset < 4096) {

        size_t want = 4096 - ioc->encinput.offset;



        buffer_reserve(&ioc->encinput, want);

        ret = qio_channel_read(ioc->master,

                               (char *)ioc->encinput.buffer +

                               ioc->encinput.offset,

                               want,

                               errp);

        if (ret < 0) {

            return ret;

        }

        if (ret == 0 &&

            ioc->encinput.offset == 0) {

            return 0;

        }

        ioc->encinput.offset += ret;

    }



    while (ioc->encinput.offset != 0) {

        if (ioc->payload_remain == 0) {

            ret = qio_channel_websock_decode_header(ioc, errp);

            if (ret < 0) {

                return ret;

            }

            if (ret == 0) {

                ioc->io_eof = TRUE;

                break;

            }

        }



        ret = qio_channel_websock_decode_payload(ioc, errp);

        if (ret < 0) {

            return ret;

        }

    }

    return 1;

}
