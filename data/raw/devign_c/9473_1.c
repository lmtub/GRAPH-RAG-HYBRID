static bool write_header(FILE *fp)

{

    static const TraceRecord header = {

        .event = HEADER_EVENT_ID,

        .timestamp_ns = HEADER_MAGIC,

        .x1 = HEADER_VERSION,

    };



    return fwrite(&header, sizeof header, 1, fp) == 1;

}
