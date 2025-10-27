int av_get_cpu_flags(void)

{

    int flags = cpu_flags;

    if (flags == -1) {

        flags = get_cpu_flags();

        cpu_flags = flags;

    }

    return flags;

}
