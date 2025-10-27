void *mcf_uart_init(qemu_irq irq, CharDriverState *chr)

{

    mcf_uart_state *s;



    s = g_malloc0(sizeof(mcf_uart_state));

    s->chr = chr;

    s->irq = irq;

    if (chr) {


        qemu_chr_add_handlers(chr, mcf_uart_can_receive, mcf_uart_receive,

                              mcf_uart_event, s);

    }

    mcf_uart_reset(s);

    return s;

}