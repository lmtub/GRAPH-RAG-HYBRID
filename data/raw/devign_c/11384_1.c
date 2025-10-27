int64_t ga_get_fd_handle(GAState *s, Error **errp)

{

    int64_t handle;



    g_assert(s->pstate_filepath);

    /* we blacklist commands and avoid operations that potentially require

     * writing to disk when we're in a frozen state. this includes opening

     * new files, so we should never get here in that situation

     */

    g_assert(!ga_is_frozen(s));



    handle = s->pstate.fd_counter++;

    if (s->pstate.fd_counter < 0) {

        s->pstate.fd_counter = 0;

    }

    if (!write_persistent_state(&s->pstate, s->pstate_filepath)) {

        error_setg(errp, "failed to commit persistent state to disk");

    }



    return handle;

}
