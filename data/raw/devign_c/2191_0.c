int qemu_aio_set_fd_handler(int fd,

                            IOHandler *io_read,

                            IOHandler *io_write,

                            AioFlushHandler *io_flush,

                            void *opaque)

{

    AioHandler *node;



    node = find_aio_handler(fd);



    /* Are we deleting the fd handler? */

    if (!io_read && !io_write) {

        if (node) {

            /* If the lock is held, just mark the node as deleted */

            if (walking_handlers)

                node->deleted = 1;

            else {

                /* Otherwise, delete it for real.  We can't just mark it as

                 * deleted because deleted nodes are only cleaned up after

                 * releasing the walking_handlers lock.

                 */

                LIST_REMOVE(node, node);

                qemu_free(node);

            }

        }

    } else {

        if (node == NULL) {

            /* Alloc and insert if it's not already there */

            node = qemu_mallocz(sizeof(AioHandler));

            node->fd = fd;

            LIST_INSERT_HEAD(&aio_handlers, node, node);

        }

        /* Update handler with latest information */

        node->io_read = io_read;

        node->io_write = io_write;

        node->io_flush = io_flush;

        node->opaque = opaque;

    }



    qemu_set_fd_handler2(fd, NULL, io_read, io_write, opaque);



    return 0;

}
