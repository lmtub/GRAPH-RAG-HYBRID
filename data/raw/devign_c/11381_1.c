void lm32_juart_set_jtx(DeviceState *d, uint32_t jtx)

{

    LM32JuartState *s = LM32_JUART(d);

    unsigned char ch = jtx & 0xff;



    trace_lm32_juart_set_jtx(s->jtx);



    s->jtx = jtx;

    if (s->chr) {



        qemu_chr_fe_write_all(s->chr, &ch, 1);

    }

}