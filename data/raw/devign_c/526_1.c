static int handle_update_file_cred(int dirfd, const char *name, FsCred *credp)

{

    int fd, ret;

    fd = openat(dirfd, name, O_NONBLOCK | O_NOFOLLOW);

    if (fd < 0) {

        return fd;

    }

    ret = fchmod(fd, credp->fc_mode & 07777);

    if (ret < 0) {

        goto err_out;

    }

    ret = fchownat(fd, "", credp->fc_uid, credp->fc_gid, AT_EMPTY_PATH);

err_out:

    close(fd);

    return ret;

}
