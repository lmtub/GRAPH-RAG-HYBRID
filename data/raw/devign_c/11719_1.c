static int scsi_hot_add(Monitor *mon, DeviceState *adapter,

                        DriveInfo *dinfo, int printinfo)

{

    SCSIBus *scsibus;

    SCSIDevice *scsidev;



    scsibus = DO_UPCAST(SCSIBus, qbus, QLIST_FIRST(&adapter->child_bus));

    if (!scsibus || strcmp(scsibus->qbus.info->name, "SCSI") != 0) {

        error_report("Device is not a SCSI adapter");





    /*

     * drive_init() tries to find a default for dinfo->unit.  Doesn't

     * work at all for hotplug though as we assign the device to a

     * specific bus instead of the first bus with spare scsi ids.

     *

     * Ditch the calculated value and reload from option string (if

     * specified).

     */

    dinfo->unit = qemu_opt_get_number(dinfo->opts, "unit", -1);

    scsidev = scsi_bus_legacy_add_drive(scsibus, dinfo, dinfo->unit);




    dinfo->unit = scsidev->id;



    if (printinfo)

        monitor_printf(mon, "OK bus %d, unit %d\n",

                       scsibus->busnr, scsidev->id);

    return 0;
