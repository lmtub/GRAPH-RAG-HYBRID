int qcow2_backing_read1(BlockDriverState *bs, QEMUIOVector *qiov,

                  int64_t sector_num, int nb_sectors)

{

    int n1;

    if ((sector_num + nb_sectors) <= bs->total_sectors)

        return nb_sectors;

    if (sector_num >= bs->total_sectors)

        n1 = 0;

    else

        n1 = bs->total_sectors - sector_num;



    qemu_iovec_memset(qiov, 0, 512 * (nb_sectors - n1));



    return n1;

}
