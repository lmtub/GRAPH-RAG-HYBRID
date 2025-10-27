static int net_socket_mcast_create(struct sockaddr_in *mcastaddr)

{

    struct ip_mreq imr;

    int fd;

    int val, ret;

    if (!IN_MULTICAST(ntohl(mcastaddr->sin_addr.s_addr))) {

	fprintf(stderr, "qemu: error: specified mcastaddr \"%s\" (0x%08x) does not contain a multicast address\n",

		inet_ntoa(mcastaddr->sin_addr),

                (int)ntohl(mcastaddr->sin_addr.s_addr));

	return -1;



    }

    fd = socket(PF_INET, SOCK_DGRAM, 0);

    if (fd < 0) {

        perror("socket(PF_INET, SOCK_DGRAM)");

        return -1;

    }



    val = 1;

    ret=setsockopt(fd, SOL_SOCKET, SO_REUSEADDR,

                   (const char *)&val, sizeof(val));

    if (ret < 0) {

	perror("setsockopt(SOL_SOCKET, SO_REUSEADDR)");

	goto fail;

    }



    ret = bind(fd, (struct sockaddr *)mcastaddr, sizeof(*mcastaddr));

    if (ret < 0) {

        perror("bind");

        goto fail;

    }



    /* Add host to multicast group */

    imr.imr_multiaddr = mcastaddr->sin_addr;

    imr.imr_interface.s_addr = htonl(INADDR_ANY);



    ret = setsockopt(fd, IPPROTO_IP, IP_ADD_MEMBERSHIP,

                     (const char *)&imr, sizeof(struct ip_mreq));

    if (ret < 0) {

	perror("setsockopt(IP_ADD_MEMBERSHIP)");

	goto fail;

    }



    /* Force mcast msgs to loopback (eg. several QEMUs in same host */

    val = 1;

    ret=setsockopt(fd, IPPROTO_IP, IP_MULTICAST_LOOP,

                   (const char *)&val, sizeof(val));

    if (ret < 0) {

	perror("setsockopt(SOL_IP, IP_MULTICAST_LOOP)");

	goto fail;

    }



    socket_set_nonblock(fd);

    return fd;

fail:

    if (fd >= 0)

        closesocket(fd);

    return -1;

}
