static void monitor_read_command(Monitor *mon, int show_prompt)

{

    if (!mon->rs)

        return;



    readline_start(mon->rs, "(qemu) ", 0, monitor_command_cb, NULL);

    if (show_prompt)

        readline_show_prompt(mon->rs);

}
