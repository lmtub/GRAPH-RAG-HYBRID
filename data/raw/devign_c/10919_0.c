int udp_output(struct socket *so, struct mbuf *m,

               struct sockaddr_in *addr)



{

    struct sockaddr_in saddr, daddr;



    saddr = *addr;

    if ((so->so_faddr.s_addr & htonl(0xffffff00)) == special_addr.s_addr) {

        if ((so->so_faddr.s_addr & htonl(0x000000ff)) == htonl(0xff))

            saddr.sin_addr.s_addr = alias_addr.s_addr;

        else if (addr->sin_addr.s_addr == loopback_addr.s_addr ||

                 ((so->so_faddr.s_addr & htonl(CTL_DNS)) == htonl(CTL_DNS)))

            saddr.sin_addr.s_addr = so->so_faddr.s_addr;

    }

    daddr.sin_addr = so->so_laddr;

    daddr.sin_port = so->so_lport;



    return udp_output2(so, m, &saddr, &daddr, so->so_iptos);

}
