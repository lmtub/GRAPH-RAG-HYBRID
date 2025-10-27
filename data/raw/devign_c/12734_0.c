int qemu_set_fd_handler(int fd,

                        IOHandler *fd_read,

                        IOHandler *fd_write,

                        void *opaque)

{

    return qemu_set_fd_handler2(fd, NULL, fd_read, fd_write, opaque);

}
