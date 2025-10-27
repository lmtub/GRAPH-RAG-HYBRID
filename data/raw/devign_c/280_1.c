void st_flush_trace_buffer(void)

{

    if (trace_file_enabled) {

        flush_trace_file();

    }



    /* Discard written trace records */

    trace_idx = 0;

}
