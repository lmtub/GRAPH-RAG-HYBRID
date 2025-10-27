void ide_bus_reset(IDEBus *bus)

{

    bus->unit = 0;

    bus->cmd = 0;

    ide_reset(&bus->ifs[0]);

    ide_reset(&bus->ifs[1]);

    ide_clear_hob(bus);



    /* pending async DMA */

    if (bus->dma->aiocb) {

#ifdef DEBUG_AIO

        printf("aio_cancel\n");

#endif

        bdrv_aio_cancel(bus->dma->aiocb);

        bus->dma->aiocb = NULL;

    }



    /* reset dma provider too */

    if (bus->dma->ops->reset) {

        bus->dma->ops->reset(bus->dma);

    }

}
