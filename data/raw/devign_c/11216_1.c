static int find_partition(BlockDriverState *bs, int partition,

                          off_t *offset, off_t *size)

{

    struct partition_record mbr[4];

    uint8_t data[512];

    int i;

    int ext_partnum = 4;



    if (bdrv_read(bs, 0, data, 1))

        errx(EINVAL, "error while reading");



    if (data[510] != 0x55 || data[511] != 0xaa) {

        errno = -EINVAL;

        return -1;

    }



    for (i = 0; i < 4; i++) {

        read_partition(&data[446 + 16 * i], &mbr[i]);



        if (!mbr[i].nb_sectors_abs)

            continue;



        if (mbr[i].system == 0xF || mbr[i].system == 0x5) {

            struct partition_record ext[4];

            uint8_t data1[512];

            int j;



            if (bdrv_read(bs, mbr[i].start_sector_abs, data1, 1))

                errx(EINVAL, "error while reading");



            for (j = 0; j < 4; j++) {

                read_partition(&data1[446 + 16 * j], &ext[j]);

                if (!ext[j].nb_sectors_abs)

                    continue;



                if ((ext_partnum + j + 1) == partition) {

                    *offset = (uint64_t)ext[j].start_sector_abs << 9;

                    *size = (uint64_t)ext[j].nb_sectors_abs << 9;

                    return 0;

                }

            }

            ext_partnum += 4;

        } else if ((i + 1) == partition) {

            *offset = (uint64_t)mbr[i].start_sector_abs << 9;

            *size = (uint64_t)mbr[i].nb_sectors_abs << 9;

            return 0;

        }

    }



    errno = -ENOENT;

    return -1;

}
