bool trace_init_backends(void)
{
#ifdef CONFIG_TRACE_SIMPLE
    if (!st_init()) {
        fprintf(stderr, "failed to initialize simple tracing backend.\n");
        return false;
    }
#ifdef CONFIG_TRACE_FTRACE
    if (!ftrace_init()) {
        fprintf(stderr, "failed to initialize ftrace backend.\n");
        return false;
    }
    return true;
}