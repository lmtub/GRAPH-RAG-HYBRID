static int scsi_get_configuration(SCSIDiskState *s, uint8_t *outbuf)

{

    int current;



    if (s->qdev.type != TYPE_ROM) {

        return -1;

    }

    current = media_is_dvd(s) ? MMC_PROFILE_DVD_ROM : MMC_PROFILE_CD_ROM;

    memset(outbuf, 0, 40);

    stl_be_p(&outbuf[0], 36); /* Bytes after the data length field */

    stw_be_p(&outbuf[6], current);

    /* outbuf[8] - outbuf[19]: Feature 0 - Profile list */

    outbuf[10] = 0x03; /* persistent, current */

    outbuf[11] = 8; /* two profiles */

    stw_be_p(&outbuf[12], MMC_PROFILE_DVD_ROM);

    outbuf[14] = (current == MMC_PROFILE_DVD_ROM);

    stw_be_p(&outbuf[16], MMC_PROFILE_CD_ROM);

    outbuf[18] = (current == MMC_PROFILE_CD_ROM);

    /* outbuf[20] - outbuf[31]: Feature 1 - Core feature */

    stw_be_p(&outbuf[20], 1);

    outbuf[22] = 0x08 | 0x03; /* version 2, persistent, current */

    outbuf[23] = 8;

    stl_be_p(&outbuf[24], 1); /* SCSI */

    outbuf[28] = 1; /* DBE = 1, mandatory */

    /* outbuf[32] - outbuf[39]: Feature 3 - Removable media feature */

    stw_be_p(&outbuf[32], 3);

    outbuf[34] = 0x08 | 0x03; /* version 2, persistent, current */

    outbuf[35] = 4;

    outbuf[36] = 0x39; /* tray, load=1, eject=1, unlocked at powerup, lock=1 */

    /* TODO: Random readable, CD read, DVD read, drive serial number,

       power management */

    return 40;

}
