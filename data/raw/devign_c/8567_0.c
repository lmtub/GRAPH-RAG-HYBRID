static void slirp_bootp_load(QEMUFile *f, Slirp *slirp)

{

    int i;



    for (i = 0; i < NB_BOOTP_CLIENTS; i++) {

        slirp->bootp_clients[i].allocated = qemu_get_be16(f);

        qemu_get_buffer(f, slirp->bootp_clients[i].macaddr, 6);

    }

}
