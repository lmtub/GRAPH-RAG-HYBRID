void vnc_client_read(void *opaque)

{

    VncState *vs = opaque;

    long ret;



#ifdef CONFIG_VNC_SASL

    if (vs->sasl.conn && vs->sasl.runSSF)

        ret = vnc_client_read_sasl(vs);

    else

#endif /* CONFIG_VNC_SASL */

#ifdef CONFIG_VNC_WS

        if (vs->encode_ws) {

            ret = vnc_client_read_ws(vs);

            if (ret == -1) {

                vnc_disconnect_start(vs);

                return;

            } else if (ret == -2) {

                vnc_client_error(vs);

                return;

            }

        } else

#endif /* CONFIG_VNC_WS */

        {

        ret = vnc_client_read_plain(vs);

        }

    if (!ret) {

        if (vs->csock == -1)

            vnc_disconnect_finish(vs);

        return;

    }



    while (vs->read_handler && vs->input.offset >= vs->read_handler_expect) {

        size_t len = vs->read_handler_expect;

        int ret;



        ret = vs->read_handler(vs, vs->input.buffer, len);

        if (vs->csock == -1) {

            vnc_disconnect_finish(vs);

            return;

        }



        if (!ret) {

            buffer_advance(&vs->input, len);

        } else {

            vs->read_handler_expect = ret;

        }

    }

}
