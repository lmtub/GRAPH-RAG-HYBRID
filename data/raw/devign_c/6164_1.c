vu_message_read(VuDev *dev, int conn_fd, VhostUserMsg *vmsg)

{

    char control[CMSG_SPACE(VHOST_MEMORY_MAX_NREGIONS * sizeof(int))] = { };

    struct iovec iov = {

        .iov_base = (char *)vmsg,

        .iov_len = VHOST_USER_HDR_SIZE,

    };

    struct msghdr msg = {

        .msg_iov = &iov,

        .msg_iovlen = 1,

        .msg_control = control,

        .msg_controllen = sizeof(control),

    };

    size_t fd_size;

    struct cmsghdr *cmsg;

    int rc;



    do {

        rc = recvmsg(conn_fd, &msg, 0);

    } while (rc < 0 && (errno == EINTR || errno == EAGAIN));



    if (rc <= 0) {

        vu_panic(dev, "Error while recvmsg: %s", strerror(errno));

        return false;

    }



    vmsg->fd_num = 0;

    for (cmsg = CMSG_FIRSTHDR(&msg);

         cmsg != NULL;

         cmsg = CMSG_NXTHDR(&msg, cmsg))

    {

        if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {

            fd_size = cmsg->cmsg_len - CMSG_LEN(0);

            vmsg->fd_num = fd_size / sizeof(int);

            memcpy(vmsg->fds, CMSG_DATA(cmsg), fd_size);

            break;

        }

    }



    if (vmsg->size > sizeof(vmsg->payload)) {

        vu_panic(dev,

                 "Error: too big message request: %d, size: vmsg->size: %u, "

                 "while sizeof(vmsg->payload) = %zu\n",

                 vmsg->request, vmsg->size, sizeof(vmsg->payload));

        goto fail;

    }



    if (vmsg->size) {

        do {

            rc = read(conn_fd, &vmsg->payload, vmsg->size);

        } while (rc < 0 && (errno == EINTR || errno == EAGAIN));



        if (rc <= 0) {

            vu_panic(dev, "Error while reading: %s", strerror(errno));

            goto fail;

        }



        assert(rc == vmsg->size);

    }



    return true;



fail:

    vmsg_close_fds(vmsg);



    return false;

}
