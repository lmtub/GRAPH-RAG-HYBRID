void monitor_init(CharDriverState *hd, int show_banner)

{

    int i;



    if (is_first_init) {

        key_timer = qemu_new_timer(vm_clock, release_keys, NULL);

        if (!key_timer)

            return;

        for (i = 0; i < MAX_MON; i++) {

            monitor_hd[i] = NULL;

        }

        is_first_init = 0;

    }

    for (i = 0; i < MAX_MON; i++) {

        if (monitor_hd[i] == NULL) {

            monitor_hd[i] = hd;

            break;

        }

    }



    hide_banner = !show_banner;



    qemu_chr_add_handlers(hd, term_can_read, term_read, term_event, NULL);



    readline_start("", 0, monitor_handle_command1, NULL);

}
