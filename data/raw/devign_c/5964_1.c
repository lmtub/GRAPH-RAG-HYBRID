sowrite(struct socket *so)

{

	int  n,nn;

	struct sbuf *sb = &so->so_rcv;

	int len = sb->sb_cc;

	struct iovec iov[2];



	DEBUG_CALL("sowrite");

	DEBUG_ARG("so = %p", so);



	if (so->so_urgc) {

		sosendoob(so);

		if (sb->sb_cc == 0)

			return 0;

	}



	/*

	 * No need to check if there's something to write,

	 * sowrite wouldn't have been called otherwise

	 */



	iov[0].iov_base = sb->sb_rptr;

        iov[1].iov_base = NULL;

        iov[1].iov_len = 0;

	if (sb->sb_rptr < sb->sb_wptr) {

		iov[0].iov_len = sb->sb_wptr - sb->sb_rptr;

		/* Should never succeed, but... */

		if (iov[0].iov_len > len) iov[0].iov_len = len;

		n = 1;

	} else {

		iov[0].iov_len = (sb->sb_data + sb->sb_datalen) - sb->sb_rptr;

		if (iov[0].iov_len > len) iov[0].iov_len = len;

		len -= iov[0].iov_len;

		if (len) {

			iov[1].iov_base = sb->sb_data;

			iov[1].iov_len = sb->sb_wptr - sb->sb_data;

			if (iov[1].iov_len > len) iov[1].iov_len = len;

			n = 2;

		} else

			n = 1;

	}

	/* Check if there's urgent data to send, and if so, send it */



#ifdef HAVE_READV

	nn = writev(so->s, (const struct iovec *)iov, n);



	DEBUG_MISC((dfd, "  ... wrote nn = %d bytes\n", nn));

#else

	nn = slirp_send(so, iov[0].iov_base, iov[0].iov_len,0);

#endif

	/* This should never happen, but people tell me it does *shrug* */

	if (nn < 0 && (errno == EAGAIN || errno == EINTR))

		return 0;



	if (nn <= 0) {

		DEBUG_MISC((dfd, " --- sowrite disconnected, so->so_state = %x, errno = %d\n",

			so->so_state, errno));

		sofcantsendmore(so);

		tcp_sockclosed(sototcpcb(so));

		return -1;

	}



#ifndef HAVE_READV

	if (n == 2 && nn == iov[0].iov_len) {

            int ret;

            ret = slirp_send(so, iov[1].iov_base, iov[1].iov_len,0);

            if (ret > 0)

                nn += ret;

        }

        DEBUG_MISC((dfd, "  ... wrote nn = %d bytes\n", nn));

#endif



	/* Update sbuf */

	sb->sb_cc -= nn;

	sb->sb_rptr += nn;

	if (sb->sb_rptr >= (sb->sb_data + sb->sb_datalen))

		sb->sb_rptr -= sb->sb_datalen;



	/*

	 * If in DRAIN mode, and there's no more data, set

	 * it CANTSENDMORE

	 */

	if ((so->so_state & SS_FWDRAIN) && sb->sb_cc == 0)

		sofcantsendmore(so);



	return nn;

}
