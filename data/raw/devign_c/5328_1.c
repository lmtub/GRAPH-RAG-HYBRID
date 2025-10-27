static int vhost_user_set_u64(struct vhost_dev *dev, int request, uint64_t u64)

{

    VhostUserMsg msg = {

        .request = request,

        .flags = VHOST_USER_VERSION,

        .u64 = u64,

        .size = sizeof(m.u64),

    };



    vhost_user_write(dev, &msg, NULL, 0);



    return 0;

}
