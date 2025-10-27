int net_init_vhost_user(const Netdev *netdev, const char *name,

                        NetClientState *peer, Error **errp)

{

    int queues;

    const NetdevVhostUserOptions *vhost_user_opts;

    CharDriverState *chr;



    assert(netdev->type == NET_CLIENT_DRIVER_VHOST_USER);

    vhost_user_opts = &netdev->u.vhost_user;



    chr = net_vhost_parse_chardev(vhost_user_opts, errp);

    if (!chr) {

        return -1;

    }



    /* verify net frontend */

    if (qemu_opts_foreach(qemu_find_opts("device"), net_vhost_check_net,

                          (char *)name, errp)) {

        return -1;

    }



    queues = vhost_user_opts->has_queues ? vhost_user_opts->queues : 1;

    if (queues < 1 || queues > MAX_QUEUE_NUM) {

        error_setg(errp,

                   "vhost-user number of queues must be in range [1, %d]",

                   MAX_QUEUE_NUM);

        return -1;

    }



    return net_vhost_user_init(peer, "vhost_user", name, chr, queues);

}
