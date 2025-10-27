static int vhost_user_set_owner(struct vhost_dev *dev)

{

    VhostUserMsg msg = {

        .request = VHOST_USER_SET_OWNER,

        .flags = VHOST_USER_VERSION,

    };



    vhost_user_write(dev, &msg, NULL, 0);



    return 0;

}
