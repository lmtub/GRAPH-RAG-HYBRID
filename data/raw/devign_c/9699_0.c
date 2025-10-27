void replay_fetch_data_kind(void)

{

    if (replay_file) {

        if (!replay_has_unread_data) {

            replay_data_kind = replay_get_byte();

            if (replay_data_kind == EVENT_INSTRUCTION) {

                replay_state.instructions_count = replay_get_dword();

            }

            replay_check_error();

            replay_has_unread_data = 1;

            if (replay_data_kind >= EVENT_COUNT) {

                error_report("Replay: unknown event kind %d", replay_data_kind);

                exit(1);

            }

        }

    }

}
