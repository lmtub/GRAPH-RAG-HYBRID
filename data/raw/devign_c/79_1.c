static i2c_interface *musicpal_audio_init(qemu_irq irq)

{

    AudioState *audio;

    musicpal_audio_state *s;

    i2c_interface *i2c;

    int iomemtype;



    audio = AUD_init();

    if (!audio) {

        AUD_log(audio_name, "No audio state\n");

        return NULL;

    }



    s = qemu_mallocz(sizeof(musicpal_audio_state));

    s->irq = irq;



    i2c = qemu_mallocz(sizeof(i2c_interface));

    i2c->bus = i2c_init_bus();

    i2c->current_addr = -1;



    s->wm = wm8750_init(i2c->bus, audio);

    if (!s->wm)

        return NULL;

    i2c_set_slave_address(s->wm, MP_WM_ADDR);

    wm8750_data_req_set(s->wm, audio_callback, s);



    iomemtype = cpu_register_io_memory(0, musicpal_audio_readfn,

                       musicpal_audio_writefn, s);

    cpu_register_physical_memory(MP_AUDIO_BASE, MP_AUDIO_SIZE, iomemtype);



    qemu_register_reset(musicpal_audio_reset, s);



    return i2c;

}
