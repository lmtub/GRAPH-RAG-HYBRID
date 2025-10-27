static int64_t qemu_icount_delta(void)

{

    if (!use_icount) {

        return 5000 * (int64_t) 1000000;

    } else if (use_icount == 1) {

        /* When not using an adaptive execution frequency

           we tend to get badly out of sync with real time,

           so just delay for a reasonable amount of time.  */

        return 0;

    } else {

        return cpu_get_icount() - cpu_get_clock();

    }

}
