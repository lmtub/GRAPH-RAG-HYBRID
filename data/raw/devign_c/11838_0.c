int nbd_init(int fd, QIOChannelSocket *sioc, uint32_t flags, off_t size)

{

    TRACE("Setting NBD socket");



    if (ioctl(fd, NBD_SET_SOCK, sioc->fd) < 0) {

        int serrno = errno;

        LOG("Failed to set NBD socket");

        return -serrno;

    }



    TRACE("Setting block size to %lu", (unsigned long)BDRV_SECTOR_SIZE);



    if (ioctl(fd, NBD_SET_BLKSIZE, (size_t)BDRV_SECTOR_SIZE) < 0) {

        int serrno = errno;

        LOG("Failed setting NBD block size");

        return -serrno;

    }



    TRACE("Setting size to %zd block(s)", (size_t)(size / BDRV_SECTOR_SIZE));



    if (ioctl(fd, NBD_SET_SIZE_BLOCKS, (size_t)(size / BDRV_SECTOR_SIZE)) < 0) {

        int serrno = errno;

        LOG("Failed setting size (in blocks)");

        return -serrno;

    }



    if (ioctl(fd, NBD_SET_FLAGS, flags) < 0) {

        if (errno == ENOTTY) {

            int read_only = (flags & NBD_FLAG_READ_ONLY) != 0;

            TRACE("Setting readonly attribute");



            if (ioctl(fd, BLKROSET, (unsigned long) &read_only) < 0) {

                int serrno = errno;

                LOG("Failed setting read-only attribute");

                return -serrno;

            }

        } else {

            int serrno = errno;

            LOG("Failed setting flags");

            return -serrno;

        }

    }



    TRACE("Negotiation ended");



    return 0;

}
