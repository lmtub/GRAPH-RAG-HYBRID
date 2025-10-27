sorecvfrom(struct socket *so)

{

	struct sockaddr_storage addr;

	socklen_t addrlen = sizeof(struct sockaddr_storage);



	DEBUG_CALL("sorecvfrom");

	DEBUG_ARG("so = %p", so);



	if (so->so_type == IPPROTO_ICMP) {   /* This is a "ping" reply */

	  char buff[256];

	  int len;



	  len = recvfrom(so->s, buff, 256, 0,

			 (struct sockaddr *)&addr, &addrlen);

	  /* XXX Check if reply is "correct"? */



	  if(len == -1 || len == 0) {

	    u_char code=ICMP_UNREACH_PORT;



	    if(errno == EHOSTUNREACH) code=ICMP_UNREACH_HOST;

	    else if(errno == ENETUNREACH) code=ICMP_UNREACH_NET;



	    DEBUG_MISC((dfd," udp icmp rx errno = %d-%s\n",

			errno,strerror(errno)));

	    icmp_error(so->so_m, ICMP_UNREACH,code, 0,strerror(errno));

	  } else {

	    icmp_reflect(so->so_m);

            so->so_m = NULL; /* Don't m_free() it again! */

	  }

	  /* No need for this socket anymore, udp_detach it */

	  udp_detach(so);

	} else {                            	/* A "normal" UDP packet */

	  struct mbuf *m;

          int len;

#ifdef _WIN32

          unsigned long n;

#else

          int n;

#endif



	  m = m_get(so->slirp);

	  if (!m) {

	      return;

	  }

	  m->m_data += IF_MAXLINKHDR;



	  /*

	   * XXX Shouldn't FIONREAD packets destined for port 53,

	   * but I don't know the max packet size for DNS lookups

	   */

	  len = M_FREEROOM(m);

	  /* if (so->so_fport != htons(53)) { */

	  ioctlsocket(so->s, FIONREAD, &n);



	  if (n > len) {

	    n = (m->m_data - m->m_dat) + m->m_len + n + 1;

	    m_inc(m, n);

	    len = M_FREEROOM(m);

	  }

	  /* } */



	  m->m_len = recvfrom(so->s, m->m_data, len, 0,

			      (struct sockaddr *)&addr, &addrlen);

	  DEBUG_MISC((dfd, " did recvfrom %d, errno = %d-%s\n",

		      m->m_len, errno,strerror(errno)));

	  if(m->m_len<0) {

	    u_char code=ICMP_UNREACH_PORT;



	    if(errno == EHOSTUNREACH) code=ICMP_UNREACH_HOST;

	    else if(errno == ENETUNREACH) code=ICMP_UNREACH_NET;



	    DEBUG_MISC((dfd," rx error, tx icmp ICMP_UNREACH:%i\n", code));

	    icmp_error(so->so_m, ICMP_UNREACH,code, 0,strerror(errno));

	    m_free(m);

	  } else {

	  /*

	   * Hack: domain name lookup will be used the most for UDP,

	   * and since they'll only be used once there's no need

	   * for the 4 minute (or whatever) timeout... So we time them

	   * out much quicker (10 seconds  for now...)

	   */

	    if (so->so_expire) {

	      if (so->so_fport == htons(53))

		so->so_expire = curtime + SO_EXPIREFAST;

	      else

		so->so_expire = curtime + SO_EXPIRE;

	    }



	    /*

	     * If this packet was destined for CTL_ADDR,

	     * make it look like that's where it came from, done by udp_output

	     */

	    switch (so->so_ffamily) {

	    case AF_INET:

	        udp_output(so, m, (struct sockaddr_in *) &addr);

	        break;

	    default:

	        break;

	    }

	  } /* rx error */

	} /* if ping packet */

}
