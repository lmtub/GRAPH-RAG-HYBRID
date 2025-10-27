void aio_set_fd_handler(AioContext *ctx,

                        int fd,

                        bool is_external,

                        IOHandler *io_read,

                        IOHandler *io_write,

                        AioPollFn *io_poll,

                        void *opaque)

{

    AioHandler *node;

    bool is_new = false;

    bool deleted = false;



    qemu_lockcnt_lock(&ctx->list_lock);



    node = find_aio_handler(ctx, fd);



    /* Are we deleting the fd handler? */

    if (!io_read && !io_write && !io_poll) {

        if (node == NULL) {

            qemu_lockcnt_unlock(&ctx->list_lock);

            return;

        }



        g_source_remove_poll(&ctx->source, &node->pfd);



        /* If the lock is held, just mark the node as deleted */

        if (qemu_lockcnt_count(&ctx->list_lock)) {

            node->deleted = 1;

            node->pfd.revents = 0;

        } else {

            /* Otherwise, delete it for real.  We can't just mark it as

             * deleted because deleted nodes are only cleaned up while

             * no one is walking the handlers list.

             */

            QLIST_REMOVE(node, node);

            deleted = true;

        }



        if (!node->io_poll) {

            ctx->poll_disable_cnt--;

        }

    } else {

        if (node == NULL) {

            /* Alloc and insert if it's not already there */

            node = g_new0(AioHandler, 1);

            node->pfd.fd = fd;

            QLIST_INSERT_HEAD_RCU(&ctx->aio_handlers, node, node);



            g_source_add_poll(&ctx->source, &node->pfd);

            is_new = true;



            ctx->poll_disable_cnt += !io_poll;

        } else {

            ctx->poll_disable_cnt += !io_poll - !node->io_poll;

        }



        /* Update handler with latest information */

        node->io_read = io_read;

        node->io_write = io_write;

        node->io_poll = io_poll;

        node->opaque = opaque;

        node->is_external = is_external;



        node->pfd.events = (io_read ? G_IO_IN | G_IO_HUP | G_IO_ERR : 0);

        node->pfd.events |= (io_write ? G_IO_OUT | G_IO_ERR : 0);

    }



    aio_epoll_update(ctx, node, is_new);

    qemu_lockcnt_unlock(&ctx->list_lock);

    aio_notify(ctx);



    if (deleted) {

        g_free(node);

    }

}
