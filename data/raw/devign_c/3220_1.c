static void replay_save_event(Event *event, int checkpoint)

{

    if (replay_mode != REPLAY_MODE_PLAY) {

        /* put the event into the file */

        replay_put_event(EVENT_ASYNC);

        replay_put_byte(checkpoint);

        replay_put_byte(event->event_kind);



        /* save event-specific data */

        switch (event->event_kind) {

        case REPLAY_ASYNC_EVENT_BH:

            replay_put_qword(event->id);

            break;

        case REPLAY_ASYNC_EVENT_INPUT:

            replay_save_input_event(event->opaque);

            break;

        case REPLAY_ASYNC_EVENT_INPUT_SYNC:

            break;

        case REPLAY_ASYNC_EVENT_CHAR_READ:

            replay_event_char_read_save(event->opaque);

            break;

        default:

            error_report("Unknown ID %d of replay event", read_event_kind);

            exit(1);

        }

    }

}
