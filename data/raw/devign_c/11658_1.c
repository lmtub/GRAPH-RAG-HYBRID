static void vnc_client_read(VncState *vs)

{

    ssize_t ret;



#ifdef CONFIG_VNC_SASL

    if (vs->sasl.conn && vs->sasl.runSSF)

        ret = vnc_client_read_sasl(vs);

    else

#endif /* CONFIG_VNC_SASL */

        ret = vnc_client_read_plain(vs);

    if (!ret) {

        if (vs->disconnecting) {

            vnc_disconnect_finish(vs);

        }

        return;

    }



    while (vs->read_handler && vs->input.offset >= vs->read_handler_expect) {

        size_t len = vs->read_handler_expect;

        int ret;



        ret = vs->read_handler(vs, vs->input.buffer, len);

        if (vs->disconnecting) {

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
