static gboolean qio_channel_yield_enter(QIOChannel *ioc,

                                        GIOCondition condition,

                                        gpointer opaque)

{

    QIOChannelYieldData *data = opaque;

    qemu_coroutine_enter(data->co, NULL);

    return FALSE;

}
