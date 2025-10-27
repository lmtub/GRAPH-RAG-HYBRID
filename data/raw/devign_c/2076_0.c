void laio_attach_aio_context(LinuxAioState *s, AioContext *new_context)

{

    s->aio_context = new_context;

    s->completion_bh = aio_bh_new(new_context, qemu_laio_completion_bh, s);

    aio_set_event_notifier(new_context, &s->e, false,

                           qemu_laio_completion_cb, NULL);

}
