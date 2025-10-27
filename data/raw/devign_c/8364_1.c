static int vhost_net_start_one(struct vhost_net *net,

                               VirtIODevice *dev)

{

    struct vhost_vring_file file = { };

    int r;



    net->dev.nvqs = 2;

    net->dev.vqs = net->vqs;



    r = vhost_dev_enable_notifiers(&net->dev, dev);

    if (r < 0) {

        goto fail_notifiers;

    }



    r = vhost_dev_start(&net->dev, dev);

    if (r < 0) {

        goto fail_start;

    }



    if (net->nc->info->poll) {

        net->nc->info->poll(net->nc, false);

    }



    if (net->nc->info->type == NET_CLIENT_DRIVER_TAP) {

        qemu_set_fd_handler(net->backend, NULL, NULL, NULL);

        file.fd = net->backend;

        for (file.index = 0; file.index < net->dev.nvqs; ++file.index) {

            const VhostOps *vhost_ops = net->dev.vhost_ops;

            r = vhost_ops->vhost_net_set_backend(&net->dev, &file);

            if (r < 0) {

                r = -errno;

                goto fail;

            }

        }

    }

    return 0;

fail:

    file.fd = -1;

    if (net->nc->info->type == NET_CLIENT_DRIVER_TAP) {

        while (file.index-- > 0) {

            const VhostOps *vhost_ops = net->dev.vhost_ops;

            int r = vhost_ops->vhost_net_set_backend(&net->dev, &file);

            assert(r >= 0);

        }

    }

    if (net->nc->info->poll) {

        net->nc->info->poll(net->nc, true);

    }

    vhost_dev_stop(&net->dev, dev);

fail_start:

    vhost_dev_disable_notifiers(&net->dev, dev);

fail_notifiers:

    return r;

}
