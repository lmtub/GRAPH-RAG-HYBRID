void av_set_cpu_flags_mask(int mask)

{

    checked       = 0;

    flags         = av_get_cpu_flags() & mask;

    checked       = 1;

}
