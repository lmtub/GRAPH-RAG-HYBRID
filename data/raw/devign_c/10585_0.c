static ioreq_t *cpu_get_ioreq(XenIOState *state)

{

    int i;

    evtchn_port_t port;



    port = xc_evtchn_pending(state->xce_handle);

    if (port == state->bufioreq_local_port) {

        timer_mod(state->buffered_io_timer,

                BUFFER_IO_MAX_DELAY + qemu_clock_get_ms(QEMU_CLOCK_REALTIME));

        return NULL;

    }



    if (port != -1) {

        for (i = 0; i < max_cpus; i++) {

            if (state->ioreq_local_port[i] == port) {

                break;

            }

        }



        if (i == max_cpus) {

            hw_error("Fatal error while trying to get io event!\n");

        }



        /* unmask the wanted port again */

        xc_evtchn_unmask(state->xce_handle, port);



        /* get the io packet from shared memory */

        state->send_vcpu = i;

        return cpu_get_ioreq_from_shared_memory(state, i);

    }



    /* read error or read nothing */

    return NULL;

}
