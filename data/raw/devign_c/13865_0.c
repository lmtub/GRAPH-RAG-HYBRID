static void show_streams(WriterContext *w, AVFormatContext *fmt_ctx)

{

    int i;

    writer_print_section_header(w, SECTION_ID_STREAMS);

    for (i = 0; i < fmt_ctx->nb_streams; i++)

        if (selected_streams[i])

            show_stream(w, fmt_ctx, i, 0);

    writer_print_section_footer(w);

}
