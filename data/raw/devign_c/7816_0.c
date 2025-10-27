int tap_open(char *ifname, int ifname_size, int *vnet_hdr,

             int vnet_hdr_required, int mq_required, Error **errp)

{

    struct ifreq ifr;

    int fd, ret;

    int len = sizeof(struct virtio_net_hdr);

    unsigned int features;



    TFR(fd = open(PATH_NET_TUN, O_RDWR));

    if (fd < 0) {

        error_setg_errno(errp, errno, "could not open %s", PATH_NET_TUN);

        return -1;

    }

    memset(&ifr, 0, sizeof(ifr));

    ifr.ifr_flags = IFF_TAP | IFF_NO_PI;



    if (ioctl(fd, TUNGETFEATURES, &features) == -1) {

        error_report("warning: TUNGETFEATURES failed: %s", strerror(errno));

        features = 0;

    }



    if (features & IFF_ONE_QUEUE) {

        ifr.ifr_flags |= IFF_ONE_QUEUE;

    }



    if (*vnet_hdr) {

        if (features & IFF_VNET_HDR) {

            *vnet_hdr = 1;

            ifr.ifr_flags |= IFF_VNET_HDR;

        } else {

            *vnet_hdr = 0;

        }



        if (vnet_hdr_required && !*vnet_hdr) {

            error_setg(errp, "vnet_hdr=1 requested, but no kernel "

                       "support for IFF_VNET_HDR available");

            close(fd);

            return -1;

        }

        /*

         * Make sure vnet header size has the default value: for a persistent

         * tap it might have been modified e.g. by another instance of qemu.

         * Ignore errors since old kernels do not support this ioctl: in this

         * case the header size implicitly has the correct value.

         */

        ioctl(fd, TUNSETVNETHDRSZ, &len);

    }



    if (mq_required) {

        if (!(features & IFF_MULTI_QUEUE)) {

            error_setg(errp, "multiqueue required, but no kernel "

                       "support for IFF_MULTI_QUEUE available");

            close(fd);

            return -1;

        } else {

            ifr.ifr_flags |= IFF_MULTI_QUEUE;

        }

    }



    if (ifname[0] != '\0')

        pstrcpy(ifr.ifr_name, IFNAMSIZ, ifname);

    else

        pstrcpy(ifr.ifr_name, IFNAMSIZ, "tap%d");

    ret = ioctl(fd, TUNSETIFF, (void *) &ifr);

    if (ret != 0) {

        if (ifname[0] != '\0') {

            error_setg_errno(errp, errno, "could not configure %s (%s)",

                             PATH_NET_TUN, ifr.ifr_name);

        } else {

            error_setg_errno(errp, errno, "could not configure %s",

                             PATH_NET_TUN);

        }

        close(fd);

        return -1;

    }

    pstrcpy(ifname, ifname_size, ifr.ifr_name);

    fcntl(fd, F_SETFL, O_NONBLOCK);

    return fd;

}
