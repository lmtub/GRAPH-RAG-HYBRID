static void monitor_readline(const char *prompt, int is_password,

                             char *buf, int buf_size)

{

    readline_start(prompt, is_password, monitor_readline_cb, NULL);

    readline_show_prompt();

    monitor_readline_buf = buf;

    monitor_readline_buf_size = buf_size;

    monitor_readline_started = 1;

    while (monitor_readline_started) {

        main_loop_wait(10);

    }

}
