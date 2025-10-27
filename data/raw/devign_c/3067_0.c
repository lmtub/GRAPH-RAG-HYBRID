static ssize_t handle_aiocb_ioctl(RawPosixAIOData *aiocb)

{

    int ret;



    ret = ioctl(aiocb->aio_fildes, aiocb->aio_ioctl_cmd, aiocb->aio_ioctl_buf);

    if (ret == -1) {

        return -errno;

    }



    /*

     * This looks weird, but the aio code only considers a request

     * successful if it has written the full number of bytes.

     *

     * Now we overload aio_nbytes as aio_ioctl_cmd for the ioctl command,

     * so in fact we return the ioctl command here to make posix_aio_read()

     * happy..

     */

    return aiocb->aio_nbytes;

}
