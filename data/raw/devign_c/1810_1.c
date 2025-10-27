static void pxa2xx_pcmcia_class_init(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);



    dc->realize = pxa2xx_pcmcia_realize;

}
