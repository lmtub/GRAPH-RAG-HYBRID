void vnc_client_read(void *opaque)

{

    VncState *vs = opaque;

    long ret;



    buffer_reserve(&vs->input, 4096);



#ifdef CONFIG_VNC_TLS

    if (vs->tls.session) {

	ret = gnutls_read(vs->tls.session, buffer_end(&vs->input), 4096);

	if (ret < 0) {

	    if (ret == GNUTLS_E_AGAIN)

		errno = EAGAIN;

	    else

		errno = EIO;

	    ret = -1;

	}

    } else

#endif /* CONFIG_VNC_TLS */

	ret = recv(vs->csock, buffer_end(&vs->input), 4096, 0);

    ret = vnc_client_io_error(vs, ret, socket_error());

    if (!ret)

	return;



    vs->input.offset += ret;



    while (vs->read_handler && vs->input.offset >= vs->read_handler_expect) {

	size_t len = vs->read_handler_expect;

	int ret;



	ret = vs->read_handler(vs, vs->input.buffer, len);

	if (vs->csock == -1)

	    return;



	if (!ret) {

	    memmove(vs->input.buffer, vs->input.buffer + len, (vs->input.offset - len));

	    vs->input.offset -= len;

	} else {

	    vs->read_handler_expect = ret;

	}

    }

}
