void qemu_aio_set_fd_handler(int fd,

                             IOHandler *io_read,

                             IOHandler *io_write,

                             AioFlushHandler *io_flush,

                             void *opaque)

{

    aio_set_fd_handler(qemu_aio_context, fd, io_read, io_write, io_flush,

                       opaque);



    qemu_set_fd_handler2(fd, NULL, io_read, io_write, opaque);

}
