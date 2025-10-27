static void sd_blk_read(SDState *sd, uint64_t addr, uint32_t len)

{

    uint64_t end = addr + len;



    DPRINTF("sd_blk_read: addr = 0x%08llx, len = %d\n",

            (unsigned long long) addr, len);

    if (!sd->bdrv || bdrv_read(sd->bdrv, addr >> 9, sd->buf, 1) < 0) {

        fprintf(stderr, "sd_blk_read: read error on host side\n");

        return;

    }



    if (end > (addr & ~511) + 512) {

        memcpy(sd->data, sd->buf + (addr & 511), 512 - (addr & 511));



        if (bdrv_read(sd->bdrv, end >> 9, sd->buf, 1) < 0) {

            fprintf(stderr, "sd_blk_read: read error on host side\n");

            return;

        }

        memcpy(sd->data + 512 - (addr & 511), sd->buf, end & 511);

    } else

        memcpy(sd->data, sd->buf + (addr & 511), len);

}
