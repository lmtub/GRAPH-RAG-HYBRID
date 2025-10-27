void i8042_setup_a20_line(ISADevice *dev, qemu_irq *a20_out)

{

    ISAKBDState *isa = I8042(dev);

    KBDState *s = &isa->kbd;



    s->a20_out = a20_out;

}
