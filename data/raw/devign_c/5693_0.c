static void vnc_client_write(void *opaque)

{

    long ret;

    VncState *vs = opaque;



#ifdef CONFIG_VNC_TLS

    if (vs->tls_session) {

	ret = gnutls_write(vs->tls_session, vs->output.buffer, vs->output.offset);

	if (ret < 0) {

	    if (ret == GNUTLS_E_AGAIN)

		errno = EAGAIN;

	    else

		errno = EIO;

	    ret = -1;

	}

    } else

#endif /* CONFIG_VNC_TLS */

	ret = send(vs->csock, vs->output.buffer, vs->output.offset, 0);

    ret = vnc_client_io_error(vs, ret, socket_error());

    if (!ret)

	return;



    memmove(vs->output.buffer, vs->output.buffer + ret, (vs->output.offset - ret));

    vs->output.offset -= ret;



    if (vs->output.offset == 0) {

	qemu_set_fd_handler2(vs->csock, NULL, vnc_client_read, NULL, vs);

    }

}
