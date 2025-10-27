static void icount_adjust_rt(void * opaque)

{

    qemu_mod_timer(icount_rt_timer,

                   qemu_get_clock(rt_clock) + 1000);

    icount_adjust();

}
