SerialState *serial_init(int base, qemu_irq irq, int baudbase,

                         CharDriverState *chr)

{

    SerialState *s;



    s = qemu_mallocz(sizeof(SerialState));

    if (!s)

        return NULL;

    s->irq = irq;

    s->baudbase = baudbase;



    s->tx_timer = qemu_new_timer(vm_clock, serial_tx_done, s);

    if (!s->tx_timer)

        return NULL;



    qemu_register_reset(serial_reset, s);

    serial_reset(s);



    register_savevm("serial", base, 2, serial_save, serial_load, s);



    register_ioport_write(base, 8, 1, serial_ioport_write, s);

    register_ioport_read(base, 8, 1, serial_ioport_read, s);

    s->chr = chr;

    qemu_chr_add_handlers(chr, serial_can_receive1, serial_receive1,

                          serial_event, s);

    return s;

}
