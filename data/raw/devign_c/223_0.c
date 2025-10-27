bool replay_next_event_is(int event)

{

    bool res = false;



    /* nothing to skip - not all instructions used */

    if (replay_state.instructions_count != 0) {

        assert(replay_data_kind == EVENT_INSTRUCTION);

        return event == EVENT_INSTRUCTION;

    }



    while (true) {

        if (event == replay_data_kind) {

            res = true;

        }

        switch (replay_data_kind) {

        case EVENT_SHUTDOWN:

            replay_finish_event();

            qemu_system_shutdown_request();

            break;

        default:

            /* clock, time_t, checkpoint and other events */

            return res;

        }

    }

    return res;

}
