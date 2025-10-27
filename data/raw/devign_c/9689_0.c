static int show_format(WriterContext *w, AVFormatContext *fmt_ctx)

{

    char val_str[128];

    int64_t size = fmt_ctx->pb ? avio_size(fmt_ctx->pb) : -1;

    int ret = 0;



    writer_print_section_header(w, SECTION_ID_FORMAT);

    print_str("filename",         fmt_ctx->filename);

    print_int("nb_streams",       fmt_ctx->nb_streams);

    print_int("nb_programs",      fmt_ctx->nb_programs);

    print_str("format_name",      fmt_ctx->iformat->name);

    if (!do_bitexact) {

        if (fmt_ctx->iformat->long_name) print_str    ("format_long_name", fmt_ctx->iformat->long_name);

        else                             print_str_opt("format_long_name", "unknown");

    }

    print_time("start_time",      fmt_ctx->start_time, &AV_TIME_BASE_Q);

    print_time("duration",        fmt_ctx->duration,   &AV_TIME_BASE_Q);

    if (size >= 0) print_val    ("size", size, unit_byte_str);

    else           print_str_opt("size", "N/A");

    if (fmt_ctx->bit_rate > 0) print_val    ("bit_rate", fmt_ctx->bit_rate, unit_bit_per_second_str);

    else                       print_str_opt("bit_rate", "N/A");

    print_int("probe_score", av_format_get_probe_score(fmt_ctx));

    ret = show_tags(w, fmt_ctx->metadata, SECTION_ID_FORMAT_TAGS);



    writer_print_section_footer(w);

    fflush(stdout);

    return ret;

}
