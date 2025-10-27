void qemu_add_balloon_handler(QEMUBalloonEvent *func, void *opaque)

{

    balloon_event_fn = func;

    balloon_opaque = opaque;

}
