static int net_init_tap_one(const NetdevTapOptions *tap, NetClientState *peer,

                            const char *model, const char *name,

                            const char *ifname, const char *script,

                            const char *downscript, const char *vhostfdname,

                            int vnet_hdr, int fd)

{

    TAPState *s;



    s = net_tap_fd_init(peer, model, name, fd, vnet_hdr);

    if (!s) {

        close(fd);

        return -1;

    }



    if (tap_set_sndbuf(s->fd, tap) < 0) {

        return -1;

    }



    if (tap->has_fd || tap->has_fds) {

        snprintf(s->nc.info_str, sizeof(s->nc.info_str), "fd=%d", fd);

    } else if (tap->has_helper) {

        snprintf(s->nc.info_str, sizeof(s->nc.info_str), "helper=%s",

                 tap->helper);

    } else {

        snprintf(s->nc.info_str, sizeof(s->nc.info_str),

                 "ifname=%s,script=%s,downscript=%s", ifname, script,

                 downscript);



        if (strcmp(downscript, "no") != 0) {

            snprintf(s->down_script, sizeof(s->down_script), "%s", downscript);

            snprintf(s->down_script_arg, sizeof(s->down_script_arg),

                     "%s", ifname);

        }

    }



    if (tap->has_vhost ? tap->vhost :

        vhostfdname || (tap->has_vhostforce && tap->vhostforce)) {

        int vhostfd;



        if (tap->has_vhostfd) {

            vhostfd = monitor_handle_fd_param(cur_mon, vhostfdname);

            if (vhostfd == -1) {

                return -1;

            }

        } else {

            vhostfd = -1;

        }



        s->vhost_net = vhost_net_init(&s->nc, vhostfd,

                                      tap->has_vhostforce && tap->vhostforce);

        if (!s->vhost_net) {

            error_report("vhost-net requested but could not be initialized");

            return -1;

        }

    } else if (tap->has_vhostfd || tap->has_vhostfds) {

        error_report("vhostfd= is not valid without vhost");

        return -1;

    }



    return 0;

}
