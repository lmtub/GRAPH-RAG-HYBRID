static CharDriverState *qemu_chr_open_win_con(const char *id,

                                              ChardevBackend *backend,

                                              ChardevReturn *ret,

                                              Error **errp)

{

    return qemu_chr_open_win_file(GetStdHandle(STD_OUTPUT_HANDLE));

}
