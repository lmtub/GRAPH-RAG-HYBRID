static int raw_write_scrubbed_bootsect(BlockDriverState *bs,

                                       const uint8_t *buf)

{

    uint8_t bootsect[512];



    /* scrub the dangerous signature */

    memcpy(bootsect, buf, 512);

    memset(bootsect, 0, 4);



    return bdrv_write(bs->file, 0, bootsect, 1);

}
