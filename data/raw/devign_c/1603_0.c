static int64_t cvtnum(const char *s)

{

    char *end;

    return qemu_strtosz_suffix(s, &end, QEMU_STRTOSZ_DEFSUFFIX_B);

}
