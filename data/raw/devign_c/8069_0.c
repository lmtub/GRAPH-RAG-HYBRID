Slirp *slirp_init(int restricted, struct in_addr vnetwork,

                  struct in_addr vnetmask, struct in_addr vhost,

                  const char *vhostname, const char *tftp_path,

                  const char *bootfile, struct in_addr vdhcp_start,

                  struct in_addr vnameserver, void *opaque)

{

    Slirp *slirp = qemu_mallocz(sizeof(Slirp));



    slirp_init_once();



    slirp->restricted = restricted;



    if_init(slirp);

    ip_init(slirp);



    /* Initialise mbufs *after* setting the MTU */

    m_init(slirp);



    slirp->vnetwork_addr = vnetwork;

    slirp->vnetwork_mask = vnetmask;

    slirp->vhost_addr = vhost;

    if (vhostname) {

        pstrcpy(slirp->client_hostname, sizeof(slirp->client_hostname),

                vhostname);

    }

    if (tftp_path) {

        slirp->tftp_prefix = qemu_strdup(tftp_path);

    }

    if (bootfile) {

        slirp->bootp_filename = qemu_strdup(bootfile);

    }

    slirp->vdhcp_startaddr = vdhcp_start;

    slirp->vnameserver_addr = vnameserver;



    slirp->opaque = opaque;



    register_savevm("slirp", 0, 3, slirp_state_save, slirp_state_load, slirp);



    TAILQ_INSERT_TAIL(&slirp_instances, slirp, entry);



    return slirp;

}
