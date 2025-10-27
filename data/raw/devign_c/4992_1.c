monitor_protocol_event_queue(MonitorEvent event,

                             QObject *data)

{

    MonitorEventState *evstate;

    int64_t now = qemu_clock_get_ns(QEMU_CLOCK_REALTIME);

    assert(event < QEVENT_MAX);



    qemu_mutex_lock(&monitor_event_state_lock);

    evstate = &(monitor_event_state[event]);

    trace_monitor_protocol_event_queue(event,

                                       data,

                                       evstate->rate,

                                       evstate->last,

                                       now);



    /* Rate limit of 0 indicates no throttling */

    if (!evstate->rate) {

        monitor_protocol_event_emit(event, data);

        evstate->last = now;

    } else {

        int64_t delta = now - evstate->last;

        if (evstate->data ||

            delta < evstate->rate) {

            /* If there's an existing event pending, replace

             * it with the new event, otherwise schedule a

             * timer for delayed emission

             */

            if (evstate->data) {

                qobject_decref(evstate->data);

            } else {

                int64_t then = evstate->last + evstate->rate;

                timer_mod_ns(evstate->timer, then);

            }

            evstate->data = data;

            qobject_incref(evstate->data);

        } else {

            monitor_protocol_event_emit(event, data);

            evstate->last = now;

        }

    }

    qemu_mutex_unlock(&monitor_event_state_lock);

}
