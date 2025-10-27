GuestExec *qmp_guest_exec(const char *path,

                       bool has_arg, strList *arg,

                       bool has_env, strList *env,

                       bool has_input_data, const char *input_data,

                       bool has_capture_output, bool capture_output,

                       Error **err)

{

    GPid pid;

    GuestExec *ge = NULL;

    GuestExecInfo *gei;

    char **argv, **envp;

    strList arglist;

    gboolean ret;

    GError *gerr = NULL;

    gint in_fd, out_fd, err_fd;

    GIOChannel *in_ch, *out_ch, *err_ch;

    GSpawnFlags flags;

    bool has_output = (has_capture_output && capture_output);

    uint8_t *input = NULL;

    size_t ninput = 0;



    arglist.value = (char *)path;

    arglist.next = has_arg ? arg : NULL;



    if (has_input_data) {

        input = qbase64_decode(input_data, -1, &ninput, err);

        if (!input) {

            return NULL;

        }

    }



    argv = guest_exec_get_args(&arglist, true);

    envp = has_env ? guest_exec_get_args(env, false) : NULL;



    flags = G_SPAWN_SEARCH_PATH | G_SPAWN_DO_NOT_REAP_CHILD;

#if GLIB_CHECK_VERSION(2, 33, 2)

    flags |= G_SPAWN_SEARCH_PATH_FROM_ENVP;

#endif

    if (!has_output) {

        flags |= G_SPAWN_STDOUT_TO_DEV_NULL | G_SPAWN_STDERR_TO_DEV_NULL;

    }



    ret = g_spawn_async_with_pipes(NULL, argv, envp, flags,

            guest_exec_task_setup, NULL, &pid, has_input_data ? &in_fd : NULL,

            has_output ? &out_fd : NULL, has_output ? &err_fd : NULL, &gerr);

    if (!ret) {

        error_setg(err, QERR_QGA_COMMAND_FAILED, gerr->message);

        g_error_free(gerr);

        goto done;

    }



    ge = g_new0(GuestExec, 1);

    ge->pid = gpid_to_int64(pid);



    gei = guest_exec_info_add(pid);

    gei->has_output = has_output;

    g_child_watch_add(pid, guest_exec_child_watch, gei);



    if (has_input_data) {

        gei->in.data = input;

        gei->in.size = ninput;

#ifdef G_OS_WIN32

        in_ch = g_io_channel_win32_new_fd(in_fd);

#else

        in_ch = g_io_channel_unix_new(in_fd);

#endif

        g_io_channel_set_encoding(in_ch, NULL, NULL);

        g_io_channel_set_buffered(in_ch, false);

        g_io_channel_set_flags(in_ch, G_IO_FLAG_NONBLOCK, NULL);


        g_io_add_watch(in_ch, G_IO_OUT, guest_exec_input_watch, &gei->in);

    }



    if (has_output) {

#ifdef G_OS_WIN32

        out_ch = g_io_channel_win32_new_fd(out_fd);

        err_ch = g_io_channel_win32_new_fd(err_fd);

#else

        out_ch = g_io_channel_unix_new(out_fd);

        err_ch = g_io_channel_unix_new(err_fd);

#endif

        g_io_channel_set_encoding(out_ch, NULL, NULL);

        g_io_channel_set_encoding(err_ch, NULL, NULL);

        g_io_channel_set_buffered(out_ch, false);

        g_io_channel_set_buffered(err_ch, false);

        g_io_channel_set_close_on_unref(out_ch, true);

        g_io_channel_set_close_on_unref(err_ch, true);

        g_io_add_watch(out_ch, G_IO_IN | G_IO_HUP,

                guest_exec_output_watch, &gei->out);

        g_io_add_watch(err_ch, G_IO_IN | G_IO_HUP,

                guest_exec_output_watch, &gei->err);

    }



done:

    g_free(argv);

    g_free(envp);



    return ge;

}