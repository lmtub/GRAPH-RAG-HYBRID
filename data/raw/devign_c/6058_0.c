static void report_config_error(const char *filename, int line_num, int log_level, int *errors, const char *fmt, ...)

{

    va_list vl;

    va_start(vl, fmt);

    av_log(NULL, log_level, "%s:%d: ", filename, line_num);

    av_vlog(NULL, log_level, fmt, vl);

    va_end(vl);



    (*errors)++;

}
