static int vmdk_open_vmdk3(BlockDriverState *bs,

                           BlockDriverState *file,

                           int flags)

{

    int ret;

    uint32_t magic;

    VMDK3Header header;

    VmdkExtent *extent;



    ret = bdrv_pread(file, sizeof(magic), &header, sizeof(header));

    if (ret < 0) {

        return ret;

    }

    extent = vmdk_add_extent(bs,

                             bs->file, false,

                             le32_to_cpu(header.disk_sectors),

                             le32_to_cpu(header.l1dir_offset) << 9,

                             0, 1 << 6, 1 << 9,

                             le32_to_cpu(header.granularity));

    ret = vmdk_init_tables(bs, extent);

    if (ret) {

        /* free extent allocated by vmdk_add_extent */

        vmdk_free_last_extent(bs);

    }

    return ret;

}
