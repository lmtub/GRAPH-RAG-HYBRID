static int vhost_user_get_vring_base(struct vhost_dev *dev,

                                     struct vhost_vring_state *ring)

{

    VhostUserMsg msg = {

        .request = VHOST_USER_GET_VRING_BASE,

        .flags = VHOST_USER_VERSION,

        .state = *ring,

        .size = sizeof(*ring),

    };



    vhost_user_write(dev, &msg, NULL, 0);



    if (vhost_user_read(dev, &msg) < 0) {

        return 0;

    }



    if (msg.request != VHOST_USER_GET_VRING_BASE) {

        error_report("Received unexpected msg type. Expected %d received %d",

                     VHOST_USER_GET_VRING_BASE, msg.request);

        return -1;

    }



    if (msg.size != sizeof(m.state)) {

        error_report("Received bad msg size.");

        return -1;

    }



    *ring = msg.state;



    return 0;

}
