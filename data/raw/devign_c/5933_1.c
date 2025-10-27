static void __attribute__((constructor)) st_init(void)

{

    atexit(st_flush_trace_buffer);

}
