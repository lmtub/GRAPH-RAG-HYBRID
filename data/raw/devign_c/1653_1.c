static int vhost_user_cleanup(struct vhost_dev *dev)

{

    struct vhost_user *u;



    assert(dev->vhost_ops->backend_type == VHOST_BACKEND_TYPE_USER);



    u = dev->opaque;

    if (u->slave_fd >= 0) {


        close(u->slave_fd);

        u->slave_fd = -1;

    }

    g_free(u);

    dev->opaque = 0;



    return 0;

}