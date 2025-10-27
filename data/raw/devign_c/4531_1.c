void st_print_trace(FILE *stream, int (*stream_printf)(FILE *stream, const char *fmt, ...))

{

    unsigned int i;



    for (i = 0; i < TRACE_BUF_LEN; i++) {

        TraceRecord record;



        if (!get_trace_record(i, &record)) {

            continue;

        }

        stream_printf(stream, "Event %" PRIu64 " : %" PRIx64 " %" PRIx64

                      " %" PRIx64 " %" PRIx64 " %" PRIx64 " %" PRIx64 "\n",

                      record.event, record.x1, record.x2,

                      record.x3, record.x4, record.x5,

                      record.x6);

    }

}
