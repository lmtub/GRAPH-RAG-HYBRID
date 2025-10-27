static void print_block_info(Monitor *mon, BlockInfo *info,

                             BlockDeviceInfo *inserted, bool verbose)

{

    ImageInfo *image_info;



    assert(!info || !info->has_inserted || info->inserted == inserted);



    if (info) {

        monitor_printf(mon, "%s", info->device);

        if (inserted && inserted->has_node_name) {

            monitor_printf(mon, " (%s)", inserted->node_name);

        }

    } else {

        assert(inserted);

        monitor_printf(mon, "%s",

                       inserted->has_node_name

                       ? inserted->node_name

                       : "<anonymous>");

    }



    if (inserted) {

        monitor_printf(mon, ": %s (%s%s%s)\n",

                       inserted->file,

                       inserted->drv,

                       inserted->ro ? ", read-only" : "",

                       inserted->encrypted ? ", encrypted" : "");

    } else {

        monitor_printf(mon, ": [not inserted]\n");

    }



    if (info) {

        if (info->has_io_status && info->io_status != BLOCK_DEVICE_IO_STATUS_OK) {

            monitor_printf(mon, "    I/O status:       %s\n",

                           BlockDeviceIoStatus_lookup[info->io_status]);

        }



        if (info->removable) {

            monitor_printf(mon, "    Removable device: %slocked, tray %s\n",

                           info->locked ? "" : "not ",

                           info->tray_open ? "open" : "closed");

        }

    }





    if (!inserted) {

        return;

    }



    monitor_printf(mon, "    Cache mode:       %s%s%s\n",

                   inserted->cache->writeback ? "writeback" : "writethrough",

                   inserted->cache->direct ? ", direct" : "",

                   inserted->cache->no_flush ? ", ignore flushes" : "");



    if (inserted->has_backing_file) {

        monitor_printf(mon,

                       "    Backing file:     %s "

                       "(chain depth: %" PRId64 ")\n",

                       inserted->backing_file,

                       inserted->backing_file_depth);

    }



    if (inserted->detect_zeroes != BLOCKDEV_DETECT_ZEROES_OPTIONS_OFF) {

        monitor_printf(mon, "    Detect zeroes:    %s\n",

                       BlockdevDetectZeroesOptions_lookup[inserted->detect_zeroes]);

    }



    if (inserted->bps  || inserted->bps_rd  || inserted->bps_wr  ||

        inserted->iops || inserted->iops_rd || inserted->iops_wr)

    {

        monitor_printf(mon, "    I/O throttling:   bps=%" PRId64

                        " bps_rd=%" PRId64  " bps_wr=%" PRId64

                        " bps_max=%" PRId64

                        " bps_rd_max=%" PRId64

                        " bps_wr_max=%" PRId64

                        " iops=%" PRId64 " iops_rd=%" PRId64

                        " iops_wr=%" PRId64

                        " iops_max=%" PRId64

                        " iops_rd_max=%" PRId64

                        " iops_wr_max=%" PRId64

                        " iops_size=%" PRId64 "\n",

                        inserted->bps,

                        inserted->bps_rd,

                        inserted->bps_wr,

                        inserted->bps_max,

                        inserted->bps_rd_max,

                        inserted->bps_wr_max,

                        inserted->iops,

                        inserted->iops_rd,

                        inserted->iops_wr,

                        inserted->iops_max,

                        inserted->iops_rd_max,

                        inserted->iops_wr_max,

                        inserted->iops_size);

    }



    /* TODO: inserted->image should never be null */

    if (verbose && inserted->image) {

        monitor_printf(mon, "\nImages:\n");

        image_info = inserted->image;

        while (1) {

                bdrv_image_info_dump((fprintf_function)monitor_printf,

                                     mon, image_info);

            if (image_info->has_backing_image) {

                image_info = image_info->backing_image;

            } else {

                break;

            }

        }

    }

}
