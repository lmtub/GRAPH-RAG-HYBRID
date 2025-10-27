void qemu_chr_info(Monitor *mon)

{

    CharDriverState *chr;



    TAILQ_FOREACH(chr, &chardevs, next) {

        monitor_printf(mon, "%s: filename=%s\n", chr->label, chr->filename);

    }

}
