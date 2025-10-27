static void posix_aio_read(void *opaque)

{

    PosixAioState *s = opaque;

    ssize_t len;



    /* read all bytes from signal pipe */

    for (;;) {

        char bytes[16];



        len = read(s->rfd, bytes, sizeof(bytes));

        if (len == -1 && errno == EINTR)

            continue; /* try again */

        if (len == sizeof(bytes))

            continue; /* more to read */

        break;

    }



    posix_aio_process_queue(s);

}
