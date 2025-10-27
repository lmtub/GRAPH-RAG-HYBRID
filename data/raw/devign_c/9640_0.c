static int ssi_sd_init(SSISlave *d)

{

    DeviceState *dev = DEVICE(d);

    ssi_sd_state *s = FROM_SSI_SLAVE(ssi_sd_state, d);

    DriveInfo *dinfo;



    s->mode = SSI_SD_CMD;

    dinfo = drive_get_next(IF_SD);

    s->sd = sd_init(dinfo ? blk_bs(blk_by_legacy_dinfo(dinfo)) : NULL, true);

    if (s->sd == NULL) {

        return -1;

    }

    register_savevm(dev, "ssi_sd", -1, 1, ssi_sd_save, ssi_sd_load, s);

    return 0;

}
