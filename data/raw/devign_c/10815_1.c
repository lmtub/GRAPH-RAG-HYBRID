static int vhost_user_set_log_base(struct vhost_dev *dev, uint64_t base,

                                   struct vhost_log *log)

{

    int fds[VHOST_MEMORY_MAX_NREGIONS];

    size_t fd_num = 0;

    bool shmfd = virtio_has_feature(dev->protocol_features,

                                    VHOST_USER_PROTOCOL_F_LOG_SHMFD);

    VhostUserMsg msg = {

        .request = VHOST_USER_SET_LOG_BASE,

        .flags = VHOST_USER_VERSION,

        .u64 = base,

        .size = sizeof(m.u64),

    };



    if (shmfd && log->fd != -1) {

        fds[fd_num++] = log->fd;

    }



    vhost_user_write(dev, &msg, fds, fd_num);



    if (shmfd) {

        msg.size = 0;

        if (vhost_user_read(dev, &msg) < 0) {

            return 0;

        }



        if (msg.request != VHOST_USER_SET_LOG_BASE) {

            error_report("Received unexpected msg type. "

                         "Expected %d received %d",

                         VHOST_USER_SET_LOG_BASE, msg.request);

            return -1;

        }

    }



    return 0;

}
