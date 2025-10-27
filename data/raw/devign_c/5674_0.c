void aio_set_event_notifier_poll(AioContext *ctx,

                                 EventNotifier *notifier,

                                 EventNotifierHandler *io_poll_begin,

                                 EventNotifierHandler *io_poll_end)

{

    aio_set_fd_poll(ctx, event_notifier_get_fd(notifier),

                    (IOHandler *)io_poll_begin,

                    (IOHandler *)io_poll_end);

}
