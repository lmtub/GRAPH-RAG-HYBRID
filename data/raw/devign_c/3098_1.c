void st_print_trace_file_status(FILE *stream, int (*stream_printf)(FILE *stream, const char *fmt, ...))

{

    stream_printf(stream, "Trace file \"%s\" %s.\n",

                  trace_file_name, trace_file_enabled ? "on" : "off");

}
