void aio_set_fd_handler(AioContext *ctx,

                        int fd,

                        bool is_external,

                        IOHandler *io_read,

                        IOHandler *io_write,

                        void *opaque)

{

    AioHandler *node;

    bool is_new = false;



    node = find_aio_handler(ctx, fd);



    /* Are we deleting the fd handler? */

    if (!io_read && !io_write) {

        if (node) {

            g_source_remove_poll(&ctx->source, &node->pfd);



            /* If the lock is held, just mark the node as deleted */

            if (ctx->walking_handlers) {

                node->deleted = 1;

                node->pfd.revents = 0;

            } else {

                /* Otherwise, delete it for real.  We can't just mark it as

                 * deleted because deleted nodes are only cleaned up after

                 * releasing the walking_handlers lock.

                 */

                QLIST_REMOVE(node, node);

                g_free(node);

            }

        }

    } else {

        if (node == NULL) {

            /* Alloc and insert if it's not already there */

            node = g_new0(AioHandler, 1);

            node->pfd.fd = fd;

            QLIST_INSERT_HEAD(&ctx->aio_handlers, node, node);



            g_source_add_poll(&ctx->source, &node->pfd);

            is_new = true;

        }

        /* Update handler with latest information */

        node->io_read = io_read;

        node->io_write = io_write;

        node->opaque = opaque;

        node->is_external = is_external;



        node->pfd.events = (io_read ? G_IO_IN | G_IO_HUP | G_IO_ERR : 0);

        node->pfd.events |= (io_write ? G_IO_OUT | G_IO_ERR : 0);

    }



    aio_epoll_update(ctx, node, is_new);

    aio_notify(ctx);

}
