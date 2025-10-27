static void monitor_data_destroy(Monitor *mon)

{








    QDECREF(mon->outbuf);

    qemu_mutex_destroy(&mon->out_lock);
