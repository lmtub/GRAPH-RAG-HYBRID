static void do_memory_save(Monitor *mon, const QDict *qdict, QObject **ret_data)

{

    FILE *f;

    uint32_t size = qdict_get_int(qdict, "size");

    const char *filename = qdict_get_str(qdict, "filename");

    target_long addr = qdict_get_int(qdict, "val");

    uint32_t l;

    CPUState *env;

    uint8_t buf[1024];



    env = mon_get_cpu();

    if (!env)

        return;



    f = fopen(filename, "wb");

    if (!f) {

        monitor_printf(mon, "could not open '%s'\n", filename);

        return;

    }

    while (size != 0) {

        l = sizeof(buf);

        if (l > size)

            l = size;

        cpu_memory_rw_debug(env, addr, buf, l, 0);

        fwrite(buf, 1, l, f);

        addr += l;

        size -= l;

    }

    fclose(f);

}
