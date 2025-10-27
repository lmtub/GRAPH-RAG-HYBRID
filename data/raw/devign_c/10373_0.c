static void vty_receive(void *opaque, const uint8_t *buf, int size)

{

    VIOsPAPRVTYDevice *dev = (VIOsPAPRVTYDevice *)opaque;

    int i;



    if ((dev->in == dev->out) && size) {

        /* toggle line to simulate edge interrupt */

        qemu_irq_pulse(dev->sdev.qirq);

    }

    for (i = 0; i < size; i++) {

        assert((dev->in - dev->out) < VTERM_BUFSIZE);

        dev->buf[dev->in++ % VTERM_BUFSIZE] = buf[i];

    }

}
