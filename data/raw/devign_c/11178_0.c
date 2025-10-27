void replay_read_events(int checkpoint)

{

    while (replay_data_kind == EVENT_ASYNC) {

        Event *event = replay_read_event(checkpoint);

        if (!event) {

            break;

        }

        replay_mutex_unlock();

        replay_run_event(event);

        replay_mutex_lock();



        g_free(event);

        replay_finish_event();

        read_event_kind = -1;

    }

}
