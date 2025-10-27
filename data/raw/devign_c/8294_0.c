static void replay_enable(const char *fname, int mode)

{

    const char *fmode = NULL;

    assert(!replay_file);



    switch (mode) {

    case REPLAY_MODE_RECORD:

        fmode = "wb";

        break;

    case REPLAY_MODE_PLAY:

        fmode = "rb";

        break;

    default:

        fprintf(stderr, "Replay: internal error: invalid replay mode\n");

        exit(1);

    }



    atexit(replay_finish);



    replay_mutex_init();



    replay_file = fopen(fname, fmode);

    if (replay_file == NULL) {

        fprintf(stderr, "Replay: open %s: %s\n", fname, strerror(errno));

        exit(1);

    }



    replay_filename = g_strdup(fname);



    replay_mode = mode;

    replay_data_kind = -1;

    replay_state.instructions_count = 0;

    replay_state.current_step = 0;



    /* skip file header for RECORD and check it for PLAY */

    if (replay_mode == REPLAY_MODE_RECORD) {

        fseek(replay_file, HEADER_SIZE, SEEK_SET);

    } else if (replay_mode == REPLAY_MODE_PLAY) {

        unsigned int version = replay_get_dword();

        if (version != REPLAY_VERSION) {

            fprintf(stderr, "Replay: invalid input log file version\n");

            exit(1);

        }

        /* go to the beginning */

        fseek(replay_file, HEADER_SIZE, SEEK_SET);

        replay_fetch_data_kind();

    }



    replay_init_events();

}
