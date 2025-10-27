static void pxa2xx_pcmcia_realize(DeviceState *dev, Error **errp)

{

    PXA2xxPCMCIAState *s = PXA2XX_PCMCIA(dev);



    pcmcia_socket_register(&s->slot);

}
