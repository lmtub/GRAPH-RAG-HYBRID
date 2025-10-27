bool qemu_file_mode_is_not_valid(const char *mode)

{

    if (mode == NULL ||

        (mode[0] != 'r' && mode[0] != 'w') ||

        mode[1] != 'b' || mode[2] != 0) {

        fprintf(stderr, "qemu_fopen: Argument validity check failed\n");

        return true;

    }



    return false;

}
