void monitor_flush(Monitor *mon)

{

    int i;

    if (term_outbuf_index > 0) {

        for (i = 0; i < MAX_MON; i++)

            if (monitor_hd[i] && monitor_hd[i]->focus == 0)

                qemu_chr_write(monitor_hd[i], term_outbuf, term_outbuf_index);

        term_outbuf_index = 0;

    }

}
