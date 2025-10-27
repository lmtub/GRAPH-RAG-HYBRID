g_malloc(size_t n_bytes)

{

    void *mem;

    __coverity_negative_sink__(n_bytes);

    mem = malloc(n_bytes == 0 ? 1 : n_bytes);

    if (!mem) __coverity_panic__();

    return mem;

}
