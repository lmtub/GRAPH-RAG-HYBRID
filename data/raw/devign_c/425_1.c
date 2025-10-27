static int vhost_set_vring_file(struct vhost_dev *dev,

                                VhostUserRequest request,

                                struct vhost_vring_file *file)

{

    int fds[VHOST_MEMORY_MAX_NREGIONS];

    size_t fd_num = 0;

    VhostUserMsg msg = {

        .request = request,

        .flags = VHOST_USER_VERSION,

        .payload.u64 = file->index & VHOST_USER_VRING_IDX_MASK,

        .size = sizeof(msg.payload.u64),

    };



    if (ioeventfd_enabled() && file->fd > 0) {

        fds[fd_num++] = file->fd;

    } else {

        msg.payload.u64 |= VHOST_USER_VRING_NOFD_MASK;

    }



    vhost_user_write(dev, &msg, fds, fd_num);



    return 0;

}
