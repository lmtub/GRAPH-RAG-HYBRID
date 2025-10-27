sbappend(struct socket *so, struct mbuf *m)

{

	int ret = 0;



	DEBUG_CALL("sbappend");

	DEBUG_ARG("so = %p", so);

	DEBUG_ARG("m = %p", m);

	DEBUG_ARG("m->m_len = %d", m->m_len);



	/* Shouldn't happen, but...  e.g. foreign host closes connection */

	if (m->m_len <= 0) {

		m_free(m);

		return;

	}



	/*

	 * If there is urgent data, call sosendoob

	 * if not all was sent, sowrite will take care of the rest

	 * (The rest of this function is just an optimisation)

	 */

	if (so->so_urgc) {

		sbappendsb(&so->so_rcv, m);

		m_free(m);

		sosendoob(so);

		return;

	}



	/*

	 * We only write if there's nothing in the buffer,

	 * ottherwise it'll arrive out of order, and hence corrupt

	 */

	if (!so->so_rcv.sb_cc)

	   ret = slirp_send(so, m->m_data, m->m_len, 0);



	if (ret <= 0) {

		/*

		 * Nothing was written

		 * It's possible that the socket has closed, but

		 * we don't need to check because if it has closed,

		 * it will be detected in the normal way by soread()

		 */

		sbappendsb(&so->so_rcv, m);

	} else if (ret != m->m_len) {

		/*

		 * Something was written, but not everything..

		 * sbappendsb the rest

		 */

		m->m_len -= ret;

		m->m_data += ret;

		sbappendsb(&so->so_rcv, m);

	} /* else */

	/* Whatever happened, we free the mbuf */

	m_free(m);

}
