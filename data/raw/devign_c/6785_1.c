static void receive_from_chr_layer(SCLPConsoleLM *scon, const uint8_t *buf,

                                   int size)

{

    assert(size == 1);



    if (*buf == '\r' || *buf == '\n') {

        scon->event.event_pending = true;

        return;

    }

    scon->buf[scon->length] = *buf;

    scon->length += 1;

    if (scon->echo) {

        qemu_chr_fe_write(scon->chr, buf, size);

    }

}
