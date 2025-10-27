static void writer_print_ts(WriterContext *wctx, const char *key, int64_t ts, int is_duration)

{

    if ((!is_duration && ts == AV_NOPTS_VALUE) || (is_duration && ts == 0)) {

        writer_print_string(wctx, key, "N/A", 1);

    } else {

        writer_print_integer(wctx, key, ts);

    }

}
