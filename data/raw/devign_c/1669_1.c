void *qemu_realloc(void *ptr, size_t size)

{

    if (!size && !allow_zero_malloc()) {

        abort();

    }

    return oom_check(realloc(ptr, size ? size : 1));

}
