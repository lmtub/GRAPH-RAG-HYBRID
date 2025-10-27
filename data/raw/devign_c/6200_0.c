void replay_shutdown_request(void)

{

    if (replay_mode == REPLAY_MODE_RECORD) {

        replay_mutex_lock();

        replay_put_event(EVENT_SHUTDOWN);

        replay_mutex_unlock();

    }

}
