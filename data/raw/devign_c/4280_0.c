void pcspk_init(PITState *pit)

{

    PCSpkState *s = &pcspk_state;



    s->pit = pit;

    register_ioport_read(0x61, 1, 1, pcspk_ioport_read, s);

    register_ioport_write(0x61, 1, 1, pcspk_ioport_write, s);

}
