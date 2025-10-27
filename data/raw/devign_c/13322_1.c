static void i82374_init(I82374State *s)

{

    DMA_init(1, NULL);

    memset(s->commands, 0, sizeof(s->commands));

}
