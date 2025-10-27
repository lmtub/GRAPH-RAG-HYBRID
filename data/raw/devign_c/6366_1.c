static void audio_init(qemu_irq *pic)

{

    struct soundhw *c;

    int audio_enabled = 0;



    for (c = soundhw; !audio_enabled && c->name; ++c) {

        audio_enabled = c->enabled;

    }



    if (audio_enabled) {

        AudioState *s;



        s = AUD_init();

        if (s) {

            for (c = soundhw; c->name; ++c) {

                if (c->enabled) {

                    if (c->isa) {

                        c->init.init_isa(s, pic);

                    }

                }

            }

        }

    }

}
