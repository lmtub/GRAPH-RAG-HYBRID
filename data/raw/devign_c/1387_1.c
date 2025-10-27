void hmp_info_block(Monitor *mon, const QDict *qdict)

{

    BlockInfoList *block_list, *info;

    ImageInfo *image_info;

    const char *device = qdict_get_try_str(qdict, "device");

    bool verbose = qdict_get_try_bool(qdict, "verbose", 0);



    block_list = qmp_query_block(NULL);



    for (info = block_list; info; info = info->next) {

        if (device && strcmp(device, info->value->device)) {

            continue;

        }



        if (info != block_list) {

            monitor_printf(mon, "\n");

        }



        monitor_printf(mon, "%s", info->value->device);

        if (info->value->has_inserted) {

            monitor_printf(mon, ": %s (%s%s%s)\n",

                           info->value->inserted->file,

                           info->value->inserted->drv,

                           info->value->inserted->ro ? ", read-only" : "",

                           info->value->inserted->encrypted ? ", encrypted" : "");

        } else {

            monitor_printf(mon, ": [not inserted]\n");

        }



        if (info->value->has_io_status && info->value->io_status != BLOCK_DEVICE_IO_STATUS_OK) {

            monitor_printf(mon, "    I/O status:       %s\n",

                           BlockDeviceIoStatus_lookup[info->value->io_status]);

        }



        if (info->value->removable) {

            monitor_printf(mon, "    Removable device: %slocked, tray %s\n",

                           info->value->locked ? "" : "not ",

                           info->value->tray_open ? "open" : "closed");

        }





        if (!info->value->has_inserted) {

            continue;

        }



        if (info->value->inserted->has_backing_file) {

            monitor_printf(mon,

                           "    Backing file:     %s "

                           "(chain depth: %" PRId64 ")\n",

                           info->value->inserted->backing_file,

                           info->value->inserted->backing_file_depth);

        }



        if (info->value->inserted->bps

            || info->value->inserted->bps_rd

            || info->value->inserted->bps_wr

            || info->value->inserted->iops

            || info->value->inserted->iops_rd

            || info->value->inserted->iops_wr)

        {

            monitor_printf(mon, "    I/O throttling:   bps=%" PRId64

                            " bps_rd=%" PRId64  " bps_wr=%" PRId64

                            " iops=%" PRId64 " iops_rd=%" PRId64

                            " iops_wr=%" PRId64 "\n",

                            info->value->inserted->bps,

                            info->value->inserted->bps_rd,

                            info->value->inserted->bps_wr,

                            info->value->inserted->iops,

                            info->value->inserted->iops_rd,

                            info->value->inserted->iops_wr);

        }



        if (verbose) {

            monitor_printf(mon, "\nImages:\n");

            image_info = info->value->inserted->image;

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



    qapi_free_BlockInfoList(block_list);

}
