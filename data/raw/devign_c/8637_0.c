static void xen_main_loop_prepare(XenIOState *state)

{

    int evtchn_fd = -1;



    if (state->xce_handle != XC_HANDLER_INITIAL_VALUE) {

        evtchn_fd = xc_evtchn_fd(state->xce_handle);

    }



    state->buffered_io_timer = timer_new_ms(QEMU_CLOCK_REALTIME, handle_buffered_io,

                                                 state);



    if (evtchn_fd != -1) {

        CPUState *cpu_state;



        DPRINTF("%s: Init cpu_by_vcpu_id\n", __func__);

        CPU_FOREACH(cpu_state) {

            DPRINTF("%s: cpu_by_vcpu_id[%d]=%p\n",

                    __func__, cpu_state->cpu_index, cpu_state);

            state->cpu_by_vcpu_id[cpu_state->cpu_index] = cpu_state;

        }

        qemu_set_fd_handler(evtchn_fd, cpu_handle_ioreq, NULL, state);

    }

}
