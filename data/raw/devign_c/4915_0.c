static void replay_add_event(ReplayAsyncEventKind event_kind,

                             void *opaque,

                             void *opaque2, uint64_t id)

{

    assert(event_kind < REPLAY_ASYNC_COUNT);



    if (!replay_file || replay_mode == REPLAY_MODE_NONE

        || !events_enabled) {

        Event e;

        e.event_kind = event_kind;

        e.opaque = opaque;

        e.opaque2 = opaque2;

        e.id = id;

        replay_run_event(&e);

        return;

    }



    Event *event = g_malloc0(sizeof(Event));

    event->event_kind = event_kind;

    event->opaque = opaque;

    event->opaque2 = opaque2;

    event->id = id;



    replay_mutex_lock();

    QTAILQ_INSERT_TAIL(&events_list, event, events);

    replay_mutex_unlock();

}
