static void quit_handler(int sig)
{
    /* if we're frozen, don't exit unless we're absolutely forced to,
     * because it's basically impossible for graceful exit to complete
     * unless all log/pid files are on unfreezable filesystems. there's
     * also a very likely chance killing the agent before unfreezing
     * the filesystems is a mistake (or will be viewed as one later).
     */
    if (ga_is_frozen(ga_state)) {
        return;
    g_debug("received signal num %d, quitting", sig);
    if (g_main_loop_is_running(ga_state->main_loop)) {
        g_main_loop_quit(ga_state->main_loop);