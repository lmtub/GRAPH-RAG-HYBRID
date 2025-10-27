static int vfio_ccw_handle_request(ORB *orb, SCSW *scsw, void *data)

{

    S390CCWDevice *cdev = data;

    VFIOCCWDevice *vcdev = DO_UPCAST(VFIOCCWDevice, cdev, cdev);

    struct ccw_io_region *region = vcdev->io_region;

    int ret;



    QEMU_BUILD_BUG_ON(sizeof(region->orb_area) != sizeof(ORB));

    QEMU_BUILD_BUG_ON(sizeof(region->scsw_area) != sizeof(SCSW));

    QEMU_BUILD_BUG_ON(sizeof(region->irb_area) != sizeof(IRB));



    memset(region, 0, sizeof(*region));



    memcpy(region->orb_area, orb, sizeof(ORB));

    memcpy(region->scsw_area, scsw, sizeof(SCSW));



again:

    ret = pwrite(vcdev->vdev.fd, region,

                 vcdev->io_region_size, vcdev->io_region_offset);

    if (ret != vcdev->io_region_size) {

        if (errno == EAGAIN) {

            goto again;

        }

        error_report("vfio-ccw: wirte I/O region failed with errno=%d", errno);

        return -errno;

    }



    return region->ret_code;

}
