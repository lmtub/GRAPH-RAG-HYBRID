int qemu_lock_fd_test(int fd, int64_t start, int64_t len, bool exclusive)

{

    int ret;

    struct flock fl = {

        .l_whence = SEEK_SET,

        .l_start  = start,

        .l_len    = len,

        .l_type   = exclusive ? F_WRLCK : F_RDLCK,

    };

    ret = fcntl(fd, QEMU_GETLK, &fl);

    if (ret == -1) {

        return -errno;

    } else {

        return fl.l_type == F_UNLCK ? 0 : -EAGAIN;

    }

}
