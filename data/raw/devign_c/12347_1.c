void AUD_vlog (const char *cap, const char *fmt, va_list ap)

{

    if (conf.log_to_monitor) {

        if (cap) {

            monitor_printf(default_mon, "%s: ", cap);

        }



        monitor_vprintf(default_mon, fmt, ap);

    }

    else {

        if (cap) {

            fprintf (stderr, "%s: ", cap);

        }



        vfprintf (stderr, fmt, ap);

    }

}
