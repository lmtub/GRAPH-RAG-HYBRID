static bool aio_epoll_try_enable(AioContext *ctx)

{

    AioHandler *node;

    struct epoll_event event;



    QLIST_FOREACH(node, &ctx->aio_handlers, node) {

        int r;

        if (node->deleted || !node->pfd.events) {

            continue;

        }

        event.events = epoll_events_from_pfd(node->pfd.events);

        event.data.ptr = node;

        r = epoll_ctl(ctx->epollfd, EPOLL_CTL_ADD, node->pfd.fd, &event);

        if (r) {

            return false;

        }

    }

    ctx->epoll_enabled = true;

    return true;

}
