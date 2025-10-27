static void rc4030_unrealize(DeviceState *dev, Error **errp)

{

    rc4030State *s = RC4030(dev);

    int i;



    timer_free(s->periodic_timer);



    address_space_destroy(&s->dma_as);

    object_unparent(OBJECT(&s->dma_tt));

    object_unparent(OBJECT(&s->dma_tt_alias));

    object_unparent(OBJECT(&s->dma_mr));

    for (i = 0; i < MAX_TL_ENTRIES; ++i) {

        memory_region_del_subregion(&s->dma_mr, &s->dma_mrs[i]);

        object_unparent(OBJECT(&s->dma_mrs[i]));

    }

}
