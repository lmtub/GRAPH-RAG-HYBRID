void ga_unset_frozen(GAState *s)

{

    if (!ga_is_frozen(s)) {

        return;

    }



    /* if we delayed creation/opening of pid/log files due to being

     * in a frozen state at start up, do it now

     */

    if (s->deferred_options.log_filepath) {

        s->log_file = fopen(s->deferred_options.log_filepath, "a");

        if (!s->log_file) {

            s->log_file = stderr;

        }

        s->deferred_options.log_filepath = NULL;

    }

    ga_enable_logging(s);

    g_warning("logging re-enabled due to filesystem unfreeze");

    if (s->deferred_options.pid_filepath) {

        if (!ga_open_pidfile(s->deferred_options.pid_filepath)) {

            g_warning("failed to create/open pid file");

        }

        s->deferred_options.pid_filepath = NULL;

    }



    /* enable all disabled, non-blacklisted commands */

    ga_enable_non_blacklisted(s->blacklist);

    s->frozen = false;

    if (!ga_delete_file(s->state_filepath_isfrozen)) {

        g_warning("unable to delete %s, fsfreeze may not function properly",

                  s->state_filepath_isfrozen);

    }

}
