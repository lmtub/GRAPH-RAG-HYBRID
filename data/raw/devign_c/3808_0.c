SerialState *serial_mm_init(MemoryRegion *address_space,

                            hwaddr base, int it_shift,

                            qemu_irq irq, int baudbase,

                            CharDriverState *chr, enum device_endian end)

{

    SerialState *s;

    Error *err = NULL;



    s = g_malloc0(sizeof(SerialState));



    s->it_shift = it_shift;

    s->irq = irq;

    s->baudbase = baudbase;

    s->chr = chr;



    serial_realize_core(s, &err);

    if (err != NULL) {

        error_report("%s", error_get_pretty(err));

        error_free(err);

        exit(1);

    }

    vmstate_register(NULL, base, &vmstate_serial, s);



    memory_region_init_io(&s->io, NULL, &serial_mm_ops[end], s,

                          "serial", 8 << it_shift);

    memory_region_add_subregion(address_space, base, &s->io);



    serial_update_msl(s);

    return s;

}
