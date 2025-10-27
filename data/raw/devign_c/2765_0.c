void replay_read_next_clock(ReplayClockKind kind)

{

    unsigned int read_kind = replay_data_kind - EVENT_CLOCK;



    assert(read_kind == kind);



    int64_t clock = replay_get_qword();



    replay_check_error();

    replay_finish_event();



    replay_state.cached_clock[read_kind] = clock;

}
