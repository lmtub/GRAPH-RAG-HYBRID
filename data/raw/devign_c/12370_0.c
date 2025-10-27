static void pmac_ide_flush(DBDMA_io *io)

{

    MACIOIDEState *m = io->opaque;



    if (m->aiocb) {

        bdrv_drain_all();

    }

}
