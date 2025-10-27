static int parallels_create(const char *filename, QemuOpts *opts, Error **errp)

{

    int64_t total_size, cl_size;

    uint8_t tmp[BDRV_SECTOR_SIZE];

    Error *local_err = NULL;

    BlockBackend *file;

    uint32_t bat_entries, bat_sectors;

    ParallelsHeader header;

    int ret;



    total_size = ROUND_UP(qemu_opt_get_size_del(opts, BLOCK_OPT_SIZE, 0),

                          BDRV_SECTOR_SIZE);

    cl_size = ROUND_UP(qemu_opt_get_size_del(opts, BLOCK_OPT_CLUSTER_SIZE,

                          DEFAULT_CLUSTER_SIZE), BDRV_SECTOR_SIZE);







    ret = bdrv_create_file(filename, opts, &local_err);

    if (ret < 0) {


        return ret;




    file = blk_new_open(filename, NULL, NULL,

                        BDRV_O_RDWR | BDRV_O_PROTOCOL, &local_err);

    if (file == NULL) {


        return -EIO;




    blk_set_allow_write_beyond_eof(file, true);



    ret = blk_truncate(file, 0);

    if (ret < 0) {

        goto exit;




    bat_entries = DIV_ROUND_UP(total_size, cl_size);

    bat_sectors = DIV_ROUND_UP(bat_entry_off(bat_entries), cl_size);

    bat_sectors = (bat_sectors *  cl_size) >> BDRV_SECTOR_BITS;



    memset(&header, 0, sizeof(header));

    memcpy(header.magic, HEADER_MAGIC2, sizeof(header.magic));

    header.version = cpu_to_le32(HEADER_VERSION);

    /* don't care much about geometry, it is not used on image level */

    header.heads = cpu_to_le32(16);

    header.cylinders = cpu_to_le32(total_size / BDRV_SECTOR_SIZE / 16 / 32);

    header.tracks = cpu_to_le32(cl_size >> BDRV_SECTOR_BITS);

    header.bat_entries = cpu_to_le32(bat_entries);

    header.nb_sectors = cpu_to_le64(DIV_ROUND_UP(total_size, BDRV_SECTOR_SIZE));

    header.data_off = cpu_to_le32(bat_sectors);



    /* write all the data */

    memset(tmp, 0, sizeof(tmp));

    memcpy(tmp, &header, sizeof(header));



    ret = blk_pwrite(file, 0, tmp, BDRV_SECTOR_SIZE, 0);

    if (ret < 0) {

        goto exit;


    ret = blk_pwrite_zeroes(file, BDRV_SECTOR_SIZE,

                            (bat_sectors - 1) << BDRV_SECTOR_BITS, 0);

    if (ret < 0) {

        goto exit;


    ret = 0;



done:

    blk_unref(file);

    return ret;



exit:

    error_setg_errno(errp, -ret, "Failed to create Parallels image");

    goto done;
