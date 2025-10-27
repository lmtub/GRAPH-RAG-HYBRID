static ssize_t vnc_tls_push(gnutls_transport_ptr_t transport,

                            const void *data,

                            size_t len) {

    struct VncState *vs = (struct VncState *)transport;

    int ret;



 retry:

    ret = send(vs->csock, data, len, 0);

    if (ret < 0) {

	if (errno == EINTR)

	    goto retry;

	return -1;

    }

    return ret;

}
