static void flush_trace_file(void)

{

    /* If the trace file is not open yet, open it now */

    if (!trace_fp) {

        trace_fp = fopen(trace_file_name, "w");

        if (!trace_fp) {

            /* Avoid repeatedly trying to open file on failure */

            trace_file_enabled = false;

            return;

        }

        write_header(trace_fp);

    }



    if (trace_fp) {

        size_t unused; /* for when fwrite(3) is declared warn_unused_result */

        unused = fwrite(trace_buf, trace_idx * sizeof(trace_buf[0]), 1, trace_fp);

    }

}
