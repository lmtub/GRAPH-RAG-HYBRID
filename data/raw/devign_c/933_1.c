void slirp_init(int restricted, struct in_addr vnetwork,

                struct in_addr vnetmask, struct in_addr vhost,

                const char *vhostname, const char *tftp_path,

                const char *bootfile, struct in_addr vdhcp_start,

                struct in_addr vnameserver)

{

#ifdef _WIN32

    WSADATA Data;



    WSAStartup(MAKEWORD(2,0), &Data);

    atexit(slirp_cleanup);

#endif



    link_up = 1;

    slirp_restrict = restricted;



    if_init();

    ip_init();



    /* Initialise mbufs *after* setting the MTU */

    m_init();



    /* set default addresses */

    inet_aton("127.0.0.1", &loopback_addr);



    if (get_dns_addr(&dns_addr) < 0) {

        dns_addr = loopback_addr;

        fprintf (stderr, "Warning: No DNS servers found\n");

    }



    vnetwork_addr = vnetwork;

    vnetwork_mask = vnetmask;

    vhost_addr = vhost;

    if (vhostname) {

        pstrcpy(slirp_hostname, sizeof(slirp_hostname), vhostname);

    }

    qemu_free(tftp_prefix);

    tftp_prefix = NULL;

    if (tftp_path) {

        tftp_prefix = qemu_strdup(tftp_path);

    }

    qemu_free(bootp_filename);

    bootp_filename = NULL;

    if (bootfile) {

        bootp_filename = qemu_strdup(bootfile);

    }

    vdhcp_startaddr = vdhcp_start;

    vnameserver_addr = vnameserver;



    getouraddr();

    register_savevm("slirp", 0, 1, slirp_state_save, slirp_state_load, NULL);

}
