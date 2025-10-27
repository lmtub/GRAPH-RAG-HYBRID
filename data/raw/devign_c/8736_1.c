void qemu_tcg_configure(QemuOpts *opts, Error **errp)

{

    const char *t = qemu_opt_get(opts, "thread");

    if (t) {

        if (strcmp(t, "multi") == 0) {

            if (TCG_OVERSIZED_GUEST) {

                error_setg(errp, "No MTTCG when guest word size > hosts");

            } else if (use_icount) {

                error_setg(errp, "No MTTCG when icount is enabled");

            } else {





                if (!check_tcg_memory_orders_compatible()) {

                    error_report("Guest expects a stronger memory ordering "

                                 "than the host provides");

                    error_printf("This may cause strange/hard to debug errors");

                }

                mttcg_enabled = true;

            }

        } else if (strcmp(t, "single") == 0) {

            mttcg_enabled = false;

        } else {

            error_setg(errp, "Invalid 'thread' setting %s", t);

        }

    } else {

        mttcg_enabled = default_mttcg_enabled();

    }

}