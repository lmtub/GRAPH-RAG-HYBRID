static void gus_realizefn (DeviceState *dev, Error **errp)

{

    ISADevice *d = ISA_DEVICE(dev);

    GUSState *s = GUS (dev);

    IsaDmaClass *k;

    struct audsettings as;



    AUD_register_card ("gus", &s->card);



    as.freq = s->freq;

    as.nchannels = 2;

    as.fmt = AUD_FMT_S16;

    as.endianness = GUS_ENDIANNESS;



    s->voice = AUD_open_out (

        &s->card,

        NULL,

        "gus",

        s,

        GUS_callback,

        &as

        );



    if (!s->voice) {

        AUD_remove_card (&s->card);

        error_setg(errp, "No voice");

        return;

    }



    s->shift = 2;

    s->samples = AUD_get_buffer_size_out (s->voice) >> s->shift;

    s->mixbuf = g_malloc0 (s->samples << s->shift);



    isa_register_portio_list (d, s->port, gus_portio_list1, s, "gus");

    isa_register_portio_list (d, (s->port + 0x100) & 0xf00,

                              gus_portio_list2, s, "gus");



    s->isa_dma = isa_get_dma(isa_bus_from_device(d), s->emu.gusdma);

    k = ISADMA_GET_CLASS(s->isa_dma);

    k->register_channel(s->isa_dma, s->emu.gusdma, GUS_read_DMA, s);

    s->emu.himemaddr = s->himem;

    s->emu.gusdatapos = s->emu.himemaddr + 1024 * 1024 + 32;

    s->emu.opaque = s;

    isa_init_irq (d, &s->pic, s->emu.gusirq);



    AUD_set_active_out (s->voice, 1);

}
