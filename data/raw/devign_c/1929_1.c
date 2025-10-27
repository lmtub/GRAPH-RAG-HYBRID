static gboolean nbd_negotiate_continue(QIOChannel *ioc,

                                       GIOCondition condition,

                                       void *opaque)

{

    qemu_coroutine_enter(opaque, NULL);

    return TRUE;

}
