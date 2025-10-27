udp_listen(port, laddr, lport, flags)

	u_int port;

	u_int32_t laddr;

	u_int lport;

	int flags;

{

	struct sockaddr_in addr;

	struct socket *so;

	int addrlen = sizeof(struct sockaddr_in), opt = 1;



	if ((so = socreate()) == NULL) {

		free(so);

		return NULL;

	}

	so->s = socket(AF_INET,SOCK_DGRAM,0);

	so->so_expire = curtime + SO_EXPIRE;

	insque(so,&udb);



	addr.sin_family = AF_INET;

	addr.sin_addr.s_addr = INADDR_ANY;

	addr.sin_port = port;



	if (bind(so->s,(struct sockaddr *)&addr, addrlen) < 0) {

		udp_detach(so);

		return NULL;

	}

	setsockopt(so->s,SOL_SOCKET,SO_REUSEADDR,(char *)&opt,sizeof(int));

/*	setsockopt(so->s,SOL_SOCKET,SO_OOBINLINE,(char *)&opt,sizeof(int)); */



	getsockname(so->s,(struct sockaddr *)&addr,&addrlen);

	so->so_fport = addr.sin_port;

	if (addr.sin_addr.s_addr == 0 || addr.sin_addr.s_addr == loopback_addr.s_addr)

	   so->so_faddr = alias_addr;

	else

	   so->so_faddr = addr.sin_addr;



	so->so_lport = lport;

	so->so_laddr.s_addr = laddr;

	if (flags != SS_FACCEPTONCE)

	   so->so_expire = 0;



	so->so_state = SS_ISFCONNECTED;



	return so;

}
