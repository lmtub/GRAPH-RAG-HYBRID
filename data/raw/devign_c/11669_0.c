static void port92_init(ISADevice *dev, qemu_irq *a20_out)

{

    Port92State *s = PORT92(dev);



    s->a20_out = a20_out;

}
