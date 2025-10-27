bool virtio_disk_is_scsi(void)

{

    if (guessed_disk_nature) {

        return (blk_cfg.blk_size  == 512);

    }

    return (blk_cfg.geometry.heads == 255)

        && (blk_cfg.geometry.sectors == 63)

        && (blk_cfg.blk_size  == 512);

}
