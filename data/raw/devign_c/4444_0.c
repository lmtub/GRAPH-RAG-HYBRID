size_t slirp_socket_can_recv(struct in_addr guest_addr, int guest_port)

{

	struct iovec iov[2];

	struct socket *so;



    if (!link_up)

        return 0;



	so = slirp_find_ctl_socket(guest_addr, guest_port);



	if (!so || so->so_state & SS_NOFDREF)

		return 0;



	if (!CONN_CANFRCV(so) || so->so_snd.sb_cc >= (so->so_snd.sb_datalen/2))

		return 0;



	return sopreprbuf(so, iov, NULL);

}
