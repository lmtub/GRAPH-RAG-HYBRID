int64_t qemu_strtosz_MiB(const char *nptr, char **end)

{

    return do_strtosz(nptr, end, 'M', 1024);

}
