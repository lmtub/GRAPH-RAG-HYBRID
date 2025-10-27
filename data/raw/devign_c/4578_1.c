static int vhost_user_get_u64(struct vhost_dev *dev, int request, uint64_t *u64)

{

    VhostUserMsg msg = {

        .request = request,

        .flags = VHOST_USER_VERSION,

    };



    if (vhost_user_one_time_request(request) && dev->vq_index != 0) {

        return 0;

    }



    vhost_user_write(dev, &msg, NULL, 0);



    if (vhost_user_read(dev, &msg) < 0) {

        return 0;

    }



    if (msg.request != request) {

        error_report("Received unexpected msg type. Expected %d received %d",

                     request, msg.request);

        return -1;

    }



    if (msg.size != sizeof(m.u64)) {

        error_report("Received bad msg size.");

        return -1;

    }



    *u64 = msg.u64;



    return 0;

}
