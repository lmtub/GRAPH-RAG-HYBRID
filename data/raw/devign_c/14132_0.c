void *qemu_mallocz(size_t size)

{

    void *ptr;

    ptr = qemu_malloc(size);

    if (!ptr)

        return NULL;

    memset(ptr, 0, size);

    return ptr;

}
