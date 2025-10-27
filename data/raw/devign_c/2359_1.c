static void monitor_event(void *opaque, int event)

{

    Monitor *mon = opaque;



    switch (event) {

    case CHR_EVENT_MUX_IN:

        qemu_mutex_lock(&mon->out_lock);

        mon->mux_out = 0;

        qemu_mutex_unlock(&mon->out_lock);

        if (mon->reset_seen) {


            monitor_resume(mon);

            monitor_flush(mon);

        } else {

            mon->suspend_cnt = 0;

        }

        break;



    case CHR_EVENT_MUX_OUT:

        if (mon->reset_seen) {

            if (mon->suspend_cnt == 0) {

                monitor_printf(mon, "\n");

            }

            monitor_flush(mon);

            monitor_suspend(mon);

        } else {

            mon->suspend_cnt++;

        }

        qemu_mutex_lock(&mon->out_lock);

        mon->mux_out = 1;

        qemu_mutex_unlock(&mon->out_lock);

        break;



    case CHR_EVENT_OPENED:

        monitor_printf(mon, "QEMU %s monitor - type 'help' for more "

                       "information\n", QEMU_VERSION);

        if (!mon->mux_out) {


            readline_show_prompt(mon->rs);

        }

        mon->reset_seen = 1;

        mon_refcount++;

        break;



    case CHR_EVENT_CLOSED:

        mon_refcount--;

        monitor_fdsets_cleanup();

        break;

    }

}