static bool cmd_write_dma(IDEState *s, uint8_t cmd)

{

    bool lba48 = (cmd == WIN_WRITEDMA_EXT);



    if (!s->bs) {

        ide_abort_command(s);

        return true;

    }



    ide_cmd_lba48_transform(s, lba48);

    ide_sector_start_dma(s, IDE_DMA_WRITE);



    s->media_changed = 1;



    return false;

}
