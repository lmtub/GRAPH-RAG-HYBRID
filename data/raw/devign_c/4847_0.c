sosendto(struct socket *so, struct mbuf *m)

{

	Slirp *slirp = so->slirp;

	int ret;

	struct sockaddr_in addr;



	DEBUG_CALL("sosendto");

	DEBUG_ARG("so = %p", so);

	DEBUG_ARG("m = %p", m);



        addr.sin_family = AF_INET;

	if ((so->so_faddr.s_addr & slirp->vnetwork_mask.s_addr) ==

	    slirp->vnetwork_addr.s_addr) {

	  /* It's an alias */

	  if (so->so_faddr.s_addr == slirp->vnameserver_addr.s_addr) {

	    if (get_dns_addr(&addr.sin_addr) < 0)

	      addr.sin_addr = loopback_addr;

	  } else {

	    addr.sin_addr = loopback_addr;

	  }

	} else

	  addr.sin_addr = so->so_faddr;

	addr.sin_port = so->so_fport;



	DEBUG_MISC((dfd, " sendto()ing, addr.sin_port=%d, addr.sin_addr.s_addr=%.16s\n", ntohs(addr.sin_port), inet_ntoa(addr.sin_addr)));



	/* Don't care what port we get */

	ret = sendto(so->s, m->m_data, m->m_len, 0,

		     (struct sockaddr *)&addr, sizeof (struct sockaddr));

	if (ret < 0)

		return -1;



	/*

	 * Kill the socket if there's no reply in 4 minutes,

	 * but only if it's an expirable socket

	 */

	if (so->so_expire)

		so->so_expire = curtime + SO_EXPIRE;

	so->so_state &= SS_PERSISTENT_MASK;

	so->so_state |= SS_ISFCONNECTED; /* So that it gets select()ed */

	return 0;

}
