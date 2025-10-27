static void vhost_net_stop_one(struct vhost_net *net,

                               VirtIODevice *dev)

{

    struct vhost_vring_file file = { .fd = -1 };



    if (net->nc->info->type == NET_CLIENT_DRIVER_TAP) {

        for (file.index = 0; file.index < net->dev.nvqs; ++file.index) {

            const VhostOps *vhost_ops = net->dev.vhost_ops;

            int r = vhost_ops->vhost_net_set_backend(&net->dev, &file);

            assert(r >= 0);

        }

    }

    if (net->nc->info->poll) {

        net->nc->info->poll(net->nc, true);

    }

    vhost_dev_stop(&net->dev, dev);

    vhost_dev_disable_notifiers(&net->dev, dev);

}
