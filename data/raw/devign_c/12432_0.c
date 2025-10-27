static void show_program(WriterContext *w, AVFormatContext *fmt_ctx, AVProgram *program)

{

    int i;



    writer_print_section_header(w, SECTION_ID_PROGRAM);

    print_int("program_id", program->id);

    print_int("program_num", program->program_num);

    print_int("nb_streams", program->nb_stream_indexes);

    print_int("pmt_pid", program->pmt_pid);

    print_int("pcr_pid", program->pcr_pid);

    print_ts("start_pts", program->start_time);

    print_time("start_time", program->start_time, &AV_TIME_BASE_Q);

    print_ts("end_pts", program->end_time);

    print_time("end_time", program->end_time, &AV_TIME_BASE_Q);

    show_tags(w, program->metadata, SECTION_ID_PROGRAM_TAGS);



    writer_print_section_header(w, SECTION_ID_PROGRAM_STREAMS);

    for (i = 0; i < program->nb_stream_indexes; i++) {

        if (selected_streams[program->stream_index[i]])

            show_stream(w, fmt_ctx, program->stream_index[i], 1);

    }

    writer_print_section_footer(w);



    writer_print_section_footer(w);

}
