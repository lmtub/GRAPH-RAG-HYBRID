int net_init_tap(const Netdev *netdev, const char *name,

                 NetClientState *peer, Error **errp)

{

    const NetdevTapOptions *tap;

    int fd, vnet_hdr = 0, i = 0, queues;

    /* for the no-fd, no-helper case */

    const char *script = NULL; /* suppress wrong "uninit'd use" gcc warning */

    const char *downscript = NULL;

    Error *err = NULL;

    const char *vhostfdname;

    char ifname[128];



    assert(netdev->type == NET_CLIENT_DRIVER_TAP);

    tap = &netdev->u.tap;

    queues = tap->has_queues ? tap->queues : 1;

    vhostfdname = tap->has_vhostfd ? tap->vhostfd : NULL;



    /* QEMU vlans does not support multiqueue tap, in this case peer is set.

     * For -netdev, peer is always NULL. */

    if (peer && (tap->has_queues || tap->has_fds || tap->has_vhostfds)) {

        error_setg(errp, "Multiqueue tap cannot be used with QEMU vlans");

        return -1;

    }



    if (tap->has_fd) {

        if (tap->has_ifname || tap->has_script || tap->has_downscript ||

            tap->has_vnet_hdr || tap->has_helper || tap->has_queues ||

            tap->has_fds || tap->has_vhostfds) {

            error_setg(errp, "ifname=, script=, downscript=, vnet_hdr=, "

                       "helper=, queues=, fds=, and vhostfds= "

                       "are invalid with fd=");

            return -1;

        }



        fd = monitor_fd_param(cur_mon, tap->fd, &err);

        if (fd == -1) {

            error_propagate(errp, err);

            return -1;

        }



        fcntl(fd, F_SETFL, O_NONBLOCK);



        vnet_hdr = tap_probe_vnet_hdr(fd);



        net_init_tap_one(tap, peer, "tap", name, NULL,

                         script, downscript,

                         vhostfdname, vnet_hdr, fd, &err);

        if (err) {

            error_propagate(errp, err);

            return -1;

        }

    } else if (tap->has_fds) {

        char **fds = g_new0(char *, MAX_TAP_QUEUES);

        char **vhost_fds = g_new0(char *, MAX_TAP_QUEUES);

        int nfds, nvhosts;



        if (tap->has_ifname || tap->has_script || tap->has_downscript ||

            tap->has_vnet_hdr || tap->has_helper || tap->has_queues ||

            tap->has_vhostfd) {

            error_setg(errp, "ifname=, script=, downscript=, vnet_hdr=, "

                       "helper=, queues=, and vhostfd= "

                       "are invalid with fds=");

            return -1;

        }



        nfds = get_fds(tap->fds, fds, MAX_TAP_QUEUES);

        if (tap->has_vhostfds) {

            nvhosts = get_fds(tap->vhostfds, vhost_fds, MAX_TAP_QUEUES);

            if (nfds != nvhosts) {

                error_setg(errp, "The number of fds passed does not match "

                           "the number of vhostfds passed");

                goto free_fail;

            }

        }



        for (i = 0; i < nfds; i++) {

            fd = monitor_fd_param(cur_mon, fds[i], &err);

            if (fd == -1) {

                error_propagate(errp, err);

                goto free_fail;

            }



            fcntl(fd, F_SETFL, O_NONBLOCK);



            if (i == 0) {

                vnet_hdr = tap_probe_vnet_hdr(fd);

            } else if (vnet_hdr != tap_probe_vnet_hdr(fd)) {

                error_setg(errp,

                           "vnet_hdr not consistent across given tap fds");

                goto free_fail;

            }



            net_init_tap_one(tap, peer, "tap", name, ifname,

                             script, downscript,

                             tap->has_vhostfds ? vhost_fds[i] : NULL,

                             vnet_hdr, fd, &err);

            if (err) {

                error_propagate(errp, err);

                goto free_fail;

            }

        }

        g_free(fds);

        g_free(vhost_fds);

        return 0;



free_fail:

        for (i = 0; i < nfds; i++) {

            g_free(fds[i]);

            g_free(vhost_fds[i]);

        }

        g_free(fds);

        g_free(vhost_fds);

        return -1;

    } else if (tap->has_helper) {

        if (tap->has_ifname || tap->has_script || tap->has_downscript ||

            tap->has_vnet_hdr || tap->has_queues || tap->has_vhostfds) {

            error_setg(errp, "ifname=, script=, downscript=, vnet_hdr=, "

                       "queues=, and vhostfds= are invalid with helper=");

            return -1;

        }



        fd = net_bridge_run_helper(tap->helper,

                                   tap->has_br ?

                                   tap->br : DEFAULT_BRIDGE_INTERFACE,

                                   errp);

        if (fd == -1) {

            return -1;

        }



        fcntl(fd, F_SETFL, O_NONBLOCK);

        vnet_hdr = tap_probe_vnet_hdr(fd);



        net_init_tap_one(tap, peer, "bridge", name, ifname,

                         script, downscript, vhostfdname,

                         vnet_hdr, fd, &err);

        if (err) {

            error_propagate(errp, err);

            close(fd);

            return -1;

        }

    } else {

        if (tap->has_vhostfds) {

            error_setg(errp, "vhostfds= is invalid if fds= wasn't specified");

            return -1;

        }

        script = tap->has_script ? tap->script : DEFAULT_NETWORK_SCRIPT;

        downscript = tap->has_downscript ? tap->downscript :

            DEFAULT_NETWORK_DOWN_SCRIPT;



        if (tap->has_ifname) {

            pstrcpy(ifname, sizeof ifname, tap->ifname);

        } else {

            ifname[0] = '\0';

        }



        for (i = 0; i < queues; i++) {

            fd = net_tap_init(tap, &vnet_hdr, i >= 1 ? "no" : script,

                              ifname, sizeof ifname, queues > 1, errp);

            if (fd == -1) {

                return -1;

            }



            if (queues > 1 && i == 0 && !tap->has_ifname) {

                if (tap_fd_get_ifname(fd, ifname)) {

                    error_setg(errp, "Fail to get ifname");

                    close(fd);

                    return -1;

                }

            }



            net_init_tap_one(tap, peer, "tap", name, ifname,

                             i >= 1 ? "no" : script,

                             i >= 1 ? "no" : downscript,

                             vhostfdname, vnet_hdr, fd, &err);

            if (err) {

                error_propagate(errp, err);

                close(fd);

                return -1;

            }

        }

    }



    return 0;

}
