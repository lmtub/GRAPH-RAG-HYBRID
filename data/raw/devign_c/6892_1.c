void *g_try_malloc0(size_t n_bytes)

{

    __coverity_negative_sink__(n_bytes);

    return calloc(1, n_bytes == 0 ? 1 : n_bytes);

}
