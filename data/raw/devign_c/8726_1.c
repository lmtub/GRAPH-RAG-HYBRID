static void virtio_scsi_locate_device(VDev *vdev)

{

    const uint16_t channel = 0; /* again, it's what QEMU does */

    uint16_t target;

    static uint8_t data[16 + 8 * 63];

    ScsiLunReport *r = (void *) data;

    ScsiDevice *sdev = vdev->scsi_device;

    int i, luns;



    /* QEMU has hardcoded channel #0 in many places.

     * If this hardcoded value is ever changed, we'll need to add code for

     * vdev->config.scsi.max_channel != 0 here.

     */

    debug_print_int("config.scsi.max_channel", vdev->config.scsi.max_channel);

    debug_print_int("config.scsi.max_target ", vdev->config.scsi.max_target);

    debug_print_int("config.scsi.max_lun    ", vdev->config.scsi.max_lun);

    debug_print_int("config.scsi.max_sectors", vdev->config.scsi.max_sectors);



    if (vdev->scsi_device_selected) {

        sdev->channel = vdev->selected_scsi_device.channel;

        sdev->target = vdev->selected_scsi_device.target;

        sdev->lun = vdev->selected_scsi_device.lun;



        IPL_check(sdev->channel == 0, "non-zero channel requested");

        IPL_check(sdev->target <= vdev->config.scsi.max_target, "target# high");

        IPL_check(sdev->lun <= vdev->config.scsi.max_lun, "LUN# high");

        return;

    }



    for (target = 0; target <= vdev->config.scsi.max_target; target++) {

        sdev->channel = channel;

        sdev->target = target; /* sdev->lun will be 0 here */

        if (!scsi_report_luns(vdev, data, sizeof(data))) {

            if (resp.response == VIRTIO_SCSI_S_BAD_TARGET) {

                continue;

            }

            print_int("target", target);

            virtio_scsi_verify_response(&resp, "SCSI cannot report LUNs");

        }

        if (r->lun_list_len == 0) {

            print_int("no LUNs for target", target);

            continue;

        }

        luns = r->lun_list_len / 8;

        debug_print_int("LUNs reported", luns);

        if (luns == 1) {

            /* There is no ",lun=#" arg for -device or ",lun=0" given.

             * Hence, the only LUN reported.

             * Usually, it's 0.

             */

            sdev->lun = r->lun[0].v16[0]; /* it's returned this way */

            debug_print_int("Have to use LUN", sdev->lun);

            return; /* we have to use this device */

        }

        for (i = 0; i < luns; i++) {

            if (r->lun[i].v64) {

                /* Look for non-zero LUN - we have where to choose from */

                sdev->lun = r->lun[i].v16[0];

                debug_print_int("Will use LUN", sdev->lun);

                return; /* we have found a device */

            }

        }

    }

    panic("\n! Cannot locate virtio-scsi device !\n");

}
