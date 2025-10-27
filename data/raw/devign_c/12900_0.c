static void icount_warp_rt(void *opaque)

{

    if (vm_clock_warp_start == -1) {

        return;

    }



    if (runstate_is_running()) {

        int64_t clock = qemu_clock_get_ns(QEMU_CLOCK_REALTIME);

        int64_t warp_delta = clock - vm_clock_warp_start;

        if (use_icount == 1) {

            qemu_icount_bias += warp_delta;

        } else {

            /*

             * In adaptive mode, do not let QEMU_CLOCK_VIRTUAL run too

             * far ahead of real time.

             */

            int64_t cur_time = cpu_get_clock();

            int64_t cur_icount = cpu_get_icount();

            int64_t delta = cur_time - cur_icount;

            qemu_icount_bias += MIN(warp_delta, delta);

        }

        if (qemu_clock_expired(QEMU_CLOCK_VIRTUAL)) {

            qemu_clock_notify(QEMU_CLOCK_VIRTUAL);

        }

    }

    vm_clock_warp_start = -1;

}
