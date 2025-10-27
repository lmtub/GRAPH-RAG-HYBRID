void ide_dma_cb(void *opaque, int ret)

{

    IDEState *s = opaque;

    int n;

    int64_t sector_num;

    bool stay_active = false;



    if (ret < 0) {

        int op = BM_STATUS_DMA_RETRY;



        if (s->dma_cmd == IDE_DMA_READ)

            op |= BM_STATUS_RETRY_READ;

        else if (s->dma_cmd == IDE_DMA_TRIM)

            op |= BM_STATUS_RETRY_TRIM;



        if (ide_handle_rw_error(s, -ret, op)) {

            return;





    n = s->io_buffer_size >> 9;

    if (n > s->nsector) {

        /* The PRDs were longer than needed for this request. Shorten them so

         * we don't get a negative remainder. The Active bit must remain set

         * after the request completes. */

        n = s->nsector;

        stay_active = true;




    sector_num = ide_get_sector(s);

    if (n > 0) {

        dma_buf_commit(s);

        sector_num += n;

        ide_set_sector(s, sector_num);

        s->nsector -= n;




    /* end of transfer ? */

    if (s->nsector == 0) {

        s->status = READY_STAT | SEEK_STAT;

        ide_set_irq(s->bus);

        goto eot;




    /* launch next transfer */

    n = s->nsector;

    s->io_buffer_index = 0;

    s->io_buffer_size = n * 512;

    if (s->bus->dma->ops->prepare_buf(s->bus->dma, ide_cmd_is_read(s)) == 0) {

        /* The PRDs were too short. Reset the Active bit, but don't raise an

         * interrupt. */

        goto eot;




#ifdef DEBUG_AIO

    printf("ide_dma_cb: sector_num=%" PRId64 " n=%d, cmd_cmd=%d\n",

           sector_num, n, s->dma_cmd);

#endif



    switch (s->dma_cmd) {

    case IDE_DMA_READ:

        s->bus->dma->aiocb = dma_bdrv_read(s->bs, &s->sg, sector_num,

                                           ide_dma_cb, s);

        break;

    case IDE_DMA_WRITE:

        s->bus->dma->aiocb = dma_bdrv_write(s->bs, &s->sg, sector_num,

                                            ide_dma_cb, s);

        break;

    case IDE_DMA_TRIM:

        s->bus->dma->aiocb = dma_bdrv_io(s->bs, &s->sg, sector_num,

                                         ide_issue_trim, ide_dma_cb, s,

                                         DMA_DIRECTION_TO_DEVICE);

        break;


    return;



eot:

    if (s->dma_cmd == IDE_DMA_READ || s->dma_cmd == IDE_DMA_WRITE) {

        bdrv_acct_done(s->bs, &s->acct);


    ide_set_inactive(s);



