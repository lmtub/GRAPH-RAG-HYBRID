bool virtio_ipl_disk_is_valid(void)

{

    return blk_cfg.blk_size && (virtio_disk_is_scsi() || virtio_disk_is_eckd());

}
