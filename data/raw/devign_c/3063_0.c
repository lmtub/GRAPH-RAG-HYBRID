struct vhost_net *vhost_net_init(VhostNetOptions *options)

{

    int r;

    bool backend_kernel = options->backend_type == VHOST_BACKEND_TYPE_KERNEL;

    struct vhost_net *net = g_malloc(sizeof *net);



    if (!options->net_backend) {

        fprintf(stderr, "vhost-net requires net backend to be setup\n");

        goto fail;

    }



    if (backend_kernel) {

        r = vhost_net_get_fd(options->net_backend);

        if (r < 0) {

            goto fail;

        }

        net->dev.backend_features = qemu_has_vnet_hdr(options->net_backend)

            ? 0 : (1 << VHOST_NET_F_VIRTIO_NET_HDR);

        net->backend = r;

    } else {

        net->dev.backend_features = 0;

        net->backend = -1;

    }

    net->nc = options->net_backend;



    net->dev.nvqs = 2;

    net->dev.vqs = net->vqs;



    r = vhost_dev_init(&net->dev, options->opaque,

                       options->backend_type, options->force);

    if (r < 0) {

        goto fail;

    }

    if (!qemu_has_vnet_hdr_len(options->net_backend,

                               sizeof(struct virtio_net_hdr_mrg_rxbuf))) {

        net->dev.features &= ~(1 << VIRTIO_NET_F_MRG_RXBUF);

    }

    if (backend_kernel) {

        if (~net->dev.features & net->dev.backend_features) {

            fprintf(stderr, "vhost lacks feature mask %" PRIu64

                   " for backend\n",

                   (uint64_t)(~net->dev.features & net->dev.backend_features));

            vhost_dev_cleanup(&net->dev);

            goto fail;

        }

    }

    /* Set sane init value. Override when guest acks. */

    vhost_net_ack_features(net, 0);

    return net;

fail:

    g_free(net);

    return NULL;

}
