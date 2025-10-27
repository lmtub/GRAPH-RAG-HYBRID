int tcp_fconnect(struct socket *so)

{

  Slirp *slirp = so->slirp;

  int ret=0;



  DEBUG_CALL("tcp_fconnect");

  DEBUG_ARG("so = %lx", (long )so);



  if( (ret=so->s=socket(AF_INET,SOCK_STREAM,0)) >= 0) {

    int opt, s=so->s;

    struct sockaddr_in addr;



    fd_nonblock(s);

    opt = 1;

    setsockopt(s,SOL_SOCKET,SO_REUSEADDR,(char *)&opt,sizeof(opt ));

    opt = 1;

    setsockopt(s,SOL_SOCKET,SO_OOBINLINE,(char *)&opt,sizeof(opt ));



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



    DEBUG_MISC((dfd, " connect()ing, addr.sin_port=%d, "

		"addr.sin_addr.s_addr=%.16s\n",

		ntohs(addr.sin_port), inet_ntoa(addr.sin_addr)));

    /* We don't care what port we get */

    ret = connect(s,(struct sockaddr *)&addr,sizeof (addr));



    /*

     * If it's not in progress, it failed, so we just return 0,

     * without clearing SS_NOFDREF

     */

    soisfconnecting(so);

  }



  return(ret);

}
