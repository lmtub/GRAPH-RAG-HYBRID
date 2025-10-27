static void monitor_protocol_event_handler(void *opaque)

{

    MonitorEventState *evstate = opaque;

    int64_t now = qemu_clock_get_ns(QEMU_CLOCK_REALTIME);



    qemu_mutex_lock(&monitor_event_state_lock);



    trace_monitor_protocol_event_handler(evstate->event,

                                         evstate->data,

                                         evstate->last,

                                         now);

    if (evstate->data) {

        monitor_protocol_event_emit(evstate->event, evstate->data);

        qobject_decref(evstate->data);

        evstate->data = NULL;

    }

    evstate->last = now;

    qemu_mutex_unlock(&monitor_event_state_lock);

}
