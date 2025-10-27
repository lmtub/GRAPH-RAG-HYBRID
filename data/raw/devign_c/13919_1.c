static void adlib_realizefn (DeviceState *dev, Error **errp)

{

    AdlibState *s = ADLIB(dev);

    PortioList *port_list = g_new(PortioList, 1);

    struct audsettings as;



    if (glob_adlib) {

        error_setg (errp, "Cannot create more than 1 adlib device");

        return;

    }

    glob_adlib = s;



#ifdef HAS_YMF262

    if (YMF262Init (1, 14318180, s->freq)) {

        error_setg (errp, "YMF262Init %d failed", s->freq);

        return;

    }

    else {

        YMF262SetTimerHandler (0, timer_handler, 0);

        s->enabled = 1;

    }

#else

    s->opl = OPLCreate (OPL_TYPE_YM3812, 3579545, s->freq);

    if (!s->opl) {

        error_setg (errp, "OPLCreate %d failed", s->freq);

        return;

    }

    else {

        OPLSetTimerHandler (s->opl, timer_handler, 0);

        s->enabled = 1;

    }

#endif



    as.freq = s->freq;

    as.nchannels = SHIFT;

    as.fmt = AUD_FMT_S16;

    as.endianness = AUDIO_HOST_ENDIANNESS;



    AUD_register_card ("adlib", &s->card);



    s->voice = AUD_open_out (

        &s->card,

        s->voice,

        "adlib",

        s,

        adlib_callback,

        &as

        );

    if (!s->voice) {

        Adlib_fini (s);

        error_setg (errp, "Initializing audio voice failed");

        return;

    }



    s->samples = AUD_get_buffer_size_out (s->voice) >> SHIFT;

    s->mixbuf = g_malloc0 (s->samples << SHIFT);



    adlib_portio_list[0].offset = s->port;

    adlib_portio_list[1].offset = s->port + 8;

    portio_list_init (port_list, OBJECT(s), adlib_portio_list, s, "adlib");

    portio_list_add (port_list, isa_address_space_io(&s->parent_obj), 0);

}
