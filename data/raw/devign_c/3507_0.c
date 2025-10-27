static int qemu_dup_flags(int fd, int flags)

{

    int ret;

    int serrno;

    int dup_flags;

    int setfl_flags;



#ifdef F_DUPFD_CLOEXEC

    ret = fcntl(fd, F_DUPFD_CLOEXEC, 0);

#else

    ret = dup(fd);

    if (ret != -1) {

        qemu_set_cloexec(ret);

    }

#endif

    if (ret == -1) {

        goto fail;

    }



    dup_flags = fcntl(ret, F_GETFL);

    if (dup_flags == -1) {

        goto fail;

    }



    if ((flags & O_SYNC) != (dup_flags & O_SYNC)) {

        errno = EINVAL;

        goto fail;

    }



    /* Set/unset flags that we can with fcntl */

    setfl_flags = O_APPEND | O_ASYNC | O_NONBLOCK;

#ifdef O_NOATIME

    setfl_flags |= O_NOATIME;

#endif

#ifdef O_DIRECT

    setfl_flags |= O_DIRECT;

#endif

    dup_flags &= ~setfl_flags;

    dup_flags |= (flags & setfl_flags);

    if (fcntl(ret, F_SETFL, dup_flags) == -1) {

        goto fail;

    }



    /* Truncate the file in the cases that open() would truncate it */

    if (flags & O_TRUNC ||

            ((flags & (O_CREAT | O_EXCL)) == (O_CREAT | O_EXCL))) {

        if (ftruncate(ret, 0) == -1) {

            goto fail;

        }

    }



    return ret;



fail:

    serrno = errno;

    if (ret != -1) {

        close(ret);

    }

    errno = serrno;

    return -1;

}
