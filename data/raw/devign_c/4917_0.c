static void unsafe_flush_warning(BDRVSSHState *s, const char *what)

{

    if (!s->unsafe_flush_warning) {

        error_report("warning: ssh server %s does not support fsync",

                     s->inet->host);

        if (what) {

            error_report("to support fsync, you need %s", what);

        }

        s->unsafe_flush_warning = true;

    }

}
