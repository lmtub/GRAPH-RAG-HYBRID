void *g_try_realloc(void *mem, size_t n_bytes)

{

    __coverity_negative_sink__(n_bytes);

    return realloc(mem, n_bytes == 0 ? 1 : n_bytes);

}
