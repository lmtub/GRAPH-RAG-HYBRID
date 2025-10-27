void *g_realloc(void *ptr, size_t size)

{

    size_t old_size, copy;

    void *new_ptr;



    if (!ptr)

        return g_malloc(size);

    old_size = *(size_t *)((char *)ptr - 16);

    copy = old_size < size ? old_size : size;

    new_ptr = g_malloc(size);

    memcpy(new_ptr, ptr, copy);

    g_free(ptr);

    return new_ptr;

}
