static int qemu_balloon_status(MonitorCompletion cb, void *opaque)

{

    if (!balloon_event_fn) {

        return 0;

    }

    balloon_event_fn(balloon_opaque, 0, cb, opaque);

    return 1;

}
