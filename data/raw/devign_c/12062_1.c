static int vhost_user_reset_device(struct vhost_dev *dev)

{

    VhostUserMsg msg = {

        .request = VHOST_USER_RESET_OWNER,

        .flags = VHOST_USER_VERSION,

    };



    vhost_user_write(dev, &msg, NULL, 0);



    return 0;

}
