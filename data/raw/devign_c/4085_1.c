int vhost_set_vring_enable(NetClientState *nc, int enable)

{

    VHostNetState *net = get_vhost_net(nc);

    const VhostOps *vhost_ops;



    nc->vring_enable = enable;



    if (!net) {

        return 0;

    }



    vhost_ops = net->dev.vhost_ops;

    if (vhost_ops->vhost_set_vring_enable) {

        return vhost_ops->vhost_set_vring_enable(&net->dev, enable);

    }



    return 0;

}
