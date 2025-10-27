udp_listen(Slirp *slirp, uint32_t haddr, u_int hport, uint32_t laddr,

           u_int lport, int flags)

{

	struct sockaddr_in addr;

	struct socket *so;

	socklen_t addrlen = sizeof(struct sockaddr_in);



	so = socreate(slirp);

	if (!so) {

	    return NULL;

	}

	so->s = qemu_socket(AF_INET,SOCK_DGRAM,0);

	so->so_expire = curtime + SO_EXPIRE;

	insque(so, &slirp->udb);



	addr.sin_family = AF_INET;

	addr.sin_addr.s_addr = haddr;

	addr.sin_port = hport;



	if (bind(so->s,(struct sockaddr *)&addr, addrlen) < 0) {

		udp_detach(so);

		return NULL;

	}

	socket_set_fast_reuse(so->s);



	getsockname(so->s,(struct sockaddr *)&addr,&addrlen);

	so->so_ffamily = AF_INET;

	so->so_fport = addr.sin_port;

	if (addr.sin_addr.s_addr == 0 ||

	    addr.sin_addr.s_addr == loopback_addr.s_addr) {

	   so->so_faddr = slirp->vhost_addr;

	} else {

	   so->so_faddr = addr.sin_addr;

	}

	so->so_lfamily = AF_INET;

	so->so_lport = lport;

	so->so_laddr.s_addr = laddr;

	if (flags != SS_FACCEPTONCE)

	   so->so_expire = 0;



	so->so_state &= SS_PERSISTENT_MASK;

	so->so_state |= SS_ISFCONNECTED | flags;



	return so;

}
