void qemu_notify_event(void)

{

    /* Write 8 bytes to be compatible with eventfd.  */

    static const uint64_t val = 1;

    ssize_t ret;



    if (io_thread_fd == -1) {

        return;

    }

    do {

        ret = write(io_thread_fd, &val, sizeof(val));

    } while (ret < 0 && errno == EINTR);



    /* EAGAIN is fine, a read must be pending.  */

    if (ret < 0 && errno != EAGAIN) {

        fprintf(stderr, "qemu_notify_event: write() failed: %s\n",

                strerror(errno));

        exit(1);

    }

}
