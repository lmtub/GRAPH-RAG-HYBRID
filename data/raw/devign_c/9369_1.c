static int console_init(SCLPEvent *event)

{

    static bool console_available;



    SCLPConsole *scon = DO_UPCAST(SCLPConsole, event, event);



    if (console_available) {

        error_report("Multiple VT220 operator consoles are not supported");

        return -1;

    }

    console_available = true;

    if (scon->chr) {

        qemu_chr_add_handlers(scon->chr, chr_can_read,

                              chr_read, NULL, scon);

    }

    scon->irq_read_vt220 = *qemu_allocate_irqs(trigger_ascii_console_data,

                                               NULL, 1);



    return 0;

}
