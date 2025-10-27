void pcnet_h_reset(void *opaque)

{

    PCNetState *s = opaque;

    int i;

    uint16_t checksum;



    /* Initialize the PROM */







    memcpy(s->prom, s->conf.macaddr.a, 6);











    s->prom[12] = s->prom[13] = 0x00;



    s->prom[14] = s->prom[15] = 0x57;



    for (i = 0,checksum = 0; i < 16; i++)

        checksum += s->prom[i];

    *(uint16_t *)&s->prom[12] = cpu_to_le16(checksum);





    s->bcr[BCR_MSRDA] = 0x0005;

    s->bcr[BCR_MSWRA] = 0x0005;

    s->bcr[BCR_MC   ] = 0x0002;

    s->bcr[BCR_LNKST] = 0x00c0;

    s->bcr[BCR_LED1 ] = 0x0084;

    s->bcr[BCR_LED2 ] = 0x0088;

    s->bcr[BCR_LED3 ] = 0x0090;

    s->bcr[BCR_FDC  ] = 0x0000;

    s->bcr[BCR_BSBC ] = 0x9001;

    s->bcr[BCR_EECAS] = 0x0002;

    s->bcr[BCR_SWS  ] = 0x0200;

    s->bcr[BCR_PLAT ] = 0xff06;



    pcnet_s_reset(s);

    pcnet_update_irq(s);

    pcnet_poll_timer(s);

}