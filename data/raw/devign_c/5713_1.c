static int vhost_user_set_vring_addr(struct vhost_dev *dev,

                                     struct vhost_vring_addr *addr)

{

    VhostUserMsg msg = {

        .request = VHOST_USER_SET_VRING_ADDR,

        .flags = VHOST_USER_VERSION,

        .payload.addr = *addr,

        .size = sizeof(msg.payload.addr),

    };



    vhost_user_write(dev, &msg, NULL, 0);



    return 0;

}
