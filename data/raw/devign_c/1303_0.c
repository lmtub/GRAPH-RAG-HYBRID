int nbd_client(int fd)

{

    int ret;

    int serrno;



    TRACE("Doing NBD loop");



    ret = ioctl(fd, NBD_DO_IT);

    if (ret == -1 && errno == EPIPE) {

        /* NBD_DO_IT normally returns EPIPE when someone has disconnected

         * the socket via NBD_DISCONNECT.  We do not want to return 1 in

         * that case.

         */

        ret = 0;

    }

    serrno = errno;



    TRACE("NBD loop returned %d: %s", ret, strerror(serrno));



    TRACE("Clearing NBD queue");

    ioctl(fd, NBD_CLEAR_QUE);



    TRACE("Clearing NBD socket");

    ioctl(fd, NBD_CLEAR_SOCK);



    errno = serrno;

    return ret;

}
