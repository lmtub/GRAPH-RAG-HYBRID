static void slirp_socket_save(QEMUFile *f, struct socket *so)

{

    qemu_put_be32(f, so->so_urgc);

    qemu_put_be16(f, so->so_ffamily);

    switch (so->so_ffamily) {

    case AF_INET:

        qemu_put_be32(f, so->so_faddr.s_addr);

        qemu_put_be16(f, so->so_fport);

        break;

    default:

        error_report(

                "so_ffamily unknown, unable to save so_faddr and so_fport\n");

    }

    qemu_put_be16(f, so->so_lfamily);

    switch (so->so_lfamily) {

    case AF_INET:

        qemu_put_be32(f, so->so_laddr.s_addr);

        qemu_put_be16(f, so->so_lport);

        break;

    default:

        error_report(

                "so_ffamily unknown, unable to save so_laddr and so_lport\n");

    }

    qemu_put_byte(f, so->so_iptos);

    qemu_put_byte(f, so->so_emu);

    qemu_put_byte(f, so->so_type);

    qemu_put_be32(f, so->so_state);

    slirp_sbuf_save(f, &so->so_rcv);

    slirp_sbuf_save(f, &so->so_snd);

    slirp_tcp_save(f, so->so_tcpcb);

}
