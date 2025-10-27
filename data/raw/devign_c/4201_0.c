static int opt_map(OptionsContext *o, const char *opt, const char *arg)

{

    StreamMap *m = NULL;

    int i, negative = 0, file_idx;

    int sync_file_idx = -1, sync_stream_idx;

    char *p, *sync;

    char *map;



    if (*arg == '-') {

        negative = 1;

        arg++;

    }

    map = av_strdup(arg);



    /* parse sync stream first, just pick first matching stream */

    if (sync = strchr(map, ',')) {

        *sync = 0;

        sync_file_idx = strtol(sync + 1, &sync, 0);

        if (sync_file_idx >= nb_input_files || sync_file_idx < 0) {

            av_log(NULL, AV_LOG_FATAL, "Invalid sync file index: %d.\n", sync_file_idx);

            exit_program(1);

        }

        if (*sync)

            sync++;

        for (i = 0; i < input_files[sync_file_idx].nb_streams; i++)

            if (check_stream_specifier(input_files[sync_file_idx].ctx,

                                       input_files[sync_file_idx].ctx->streams[i], sync) == 1) {

                sync_stream_idx = i;

                break;

            }

        if (i == input_files[sync_file_idx].nb_streams) {

            av_log(NULL, AV_LOG_FATAL, "Sync stream specification in map %s does not "

                                       "match any streams.\n", arg);

            exit_program(1);

        }

    }





    file_idx = strtol(map, &p, 0);

    if (file_idx >= nb_input_files || file_idx < 0) {

        av_log(NULL, AV_LOG_FATAL, "Invalid input file index: %d.\n", file_idx);

        exit_program(1);

    }

    if (negative)

        /* disable some already defined maps */

        for (i = 0; i < o->nb_stream_maps; i++) {

            m = &o->stream_maps[i];

            if (check_stream_specifier(input_files[m->file_index].ctx,

                                       input_files[m->file_index].ctx->streams[m->stream_index],

                                       *p == ':' ? p + 1 : p) > 0)

                m->disabled = 1;

        }

    else

        for (i = 0; i < input_files[file_idx].nb_streams; i++) {

            if (check_stream_specifier(input_files[file_idx].ctx, input_files[file_idx].ctx->streams[i],

                        *p == ':' ? p + 1 : p) <= 0)

                continue;

            o->stream_maps = grow_array(o->stream_maps, sizeof(*o->stream_maps),

                                        &o->nb_stream_maps, o->nb_stream_maps + 1);

            m = &o->stream_maps[o->nb_stream_maps - 1];



            m->file_index   = file_idx;

            m->stream_index = i;



            if (sync_file_idx >= 0) {

                m->sync_file_index   = sync_file_idx;

                m->sync_stream_index = sync_stream_idx;

            } else {

                m->sync_file_index   = file_idx;

                m->sync_stream_index = i;

            }

        }



    if (!m) {

        av_log(NULL, AV_LOG_FATAL, "Stream map '%s' matches no streams.\n", arg);

        exit_program(1);

    }



    av_freep(&map);

    return 0;

}
