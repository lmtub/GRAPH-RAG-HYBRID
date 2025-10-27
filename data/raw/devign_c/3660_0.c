static int qemu_balloon(ram_addr_t target)

{

    if (!balloon_event_fn) {

        return 0;

    }

    trace_balloon_event(balloon_opaque, target);

    balloon_event_fn(balloon_opaque, target);

    return 1;

}
