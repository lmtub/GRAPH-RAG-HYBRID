void *g_realloc(void * mem, size_t n_bytes)

{

    __coverity_negative_sink__(n_bytes);

    mem = realloc(mem, n_bytes == 0 ? 1 : n_bytes);

    if (!mem) __coverity_panic__();

    return mem;

}
