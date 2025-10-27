int net_init_socket(QemuOpts *opts,

                    Monitor *mon,

                    const char *name,

                    VLANState *vlan)

{

    if (qemu_opt_get(opts, "fd")) {

        int fd;



        if (qemu_opt_get(opts, "listen") ||

            qemu_opt_get(opts, "connect") ||

            qemu_opt_get(opts, "mcast")) {

            qemu_error("listen=, connect= and mcast= is invalid with fd=\n");

            return -1;

        }



        fd = net_handle_fd_param(mon, qemu_opt_get(opts, "fd"));

        if (fd == -1) {

            return -1;

        }



        if (!net_socket_fd_init(vlan, "socket", name, fd, 1)) {

            close(fd);

            return -1;

        }

    } else if (qemu_opt_get(opts, "listen")) {

        const char *listen;



        if (qemu_opt_get(opts, "fd") ||

            qemu_opt_get(opts, "connect") ||

            qemu_opt_get(opts, "mcast")) {

            qemu_error("fd=, connect= and mcast= is invalid with listen=\n");

            return -1;

        }



        listen = qemu_opt_get(opts, "listen");



        if (net_socket_listen_init(vlan, "socket", name, listen) == -1) {

            return -1;

        }

    } else if (qemu_opt_get(opts, "connect")) {

        const char *connect;



        if (qemu_opt_get(opts, "fd") ||

            qemu_opt_get(opts, "listen") ||

            qemu_opt_get(opts, "mcast")) {

            qemu_error("fd=, listen= and mcast= is invalid with connect=\n");

            return -1;

        }



        connect = qemu_opt_get(opts, "connect");



        if (net_socket_connect_init(vlan, "socket", name, connect) == -1) {

            return -1;

        }

    } else if (qemu_opt_get(opts, "mcast")) {

        const char *mcast;



        if (qemu_opt_get(opts, "fd") ||

            qemu_opt_get(opts, "connect") ||

            qemu_opt_get(opts, "listen")) {

            qemu_error("fd=, connect= and listen= is invalid with mcast=\n");

            return -1;

        }



        mcast = qemu_opt_get(opts, "mcast");



        if (net_socket_mcast_init(vlan, "socket", name, mcast) == -1) {

            return -1;

        }

    } else {

        qemu_error("-socket requires fd=, listen=, connect= or mcast=\n");

        return -1;

    }



    if (vlan) {

        vlan->nb_host_devs++;

    }



    return 0;

}
