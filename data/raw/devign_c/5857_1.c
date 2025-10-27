static void ide_atapi_cmd(IDEState *s)
{
    const uint8_t *packet;
    uint8_t *buf;
    int max_len;
    packet = s->io_buffer;
    buf = s->io_buffer;
#ifdef DEBUG_IDE_ATAPI
    {
        int i;
        printf("ATAPI limit=0x%x packet:", s->lcyl | (s->hcyl << 8));
        for(i = 0; i < ATAPI_PACKET_SIZE; i++) {
            printf(" %02x", packet[i]);
        printf("\n");
#endif
    /* If there's a UNIT_ATTENTION condition pending, only
       REQUEST_SENSE and INQUIRY commands are allowed to complete. */
    if (s->sense_key == SENSE_UNIT_ATTENTION &&
	s->io_buffer[0] != GPCMD_REQUEST_SENSE &&
	s->io_buffer[0] != GPCMD_INQUIRY) {
	ide_atapi_cmd_check_status(s);
	return;
    switch(s->io_buffer[0]) {
    case GPCMD_TEST_UNIT_READY:
        if (bdrv_is_inserted(s->bs) && !s->cdrom_changed) {
            ide_atapi_cmd_ok(s);
        } else {
            s->cdrom_changed = 0;
            ide_atapi_cmd_error(s, SENSE_NOT_READY,
                                ASC_MEDIUM_NOT_PRESENT);
    case GPCMD_MODE_SENSE_6:
    case GPCMD_MODE_SENSE_10:
        {
            int action, code;
            if (packet[0] == GPCMD_MODE_SENSE_10)
            else
                max_len = packet[4];
            action = packet[2] >> 6;
            code = packet[2] & 0x3f;
            switch(action) {
            case 0: /* current values */
                switch(code) {
                case GPMODE_R_W_ERROR_PAGE: /* error recovery */
                    cpu_to_ube16(&buf[0], 16 + 6);
                    buf[2] = 0x70;
                    buf[3] = 0;
                    buf[4] = 0;
                    buf[5] = 0;
                    buf[6] = 0;
                    buf[7] = 0;
                    buf[8] = 0x01;
                    buf[9] = 0x06;
                    buf[10] = 0x00;
                    buf[11] = 0x05;
                    buf[12] = 0x00;
                    buf[13] = 0x00;
                    buf[14] = 0x00;
                    buf[15] = 0x00;
                    ide_atapi_cmd_reply(s, 16, max_len);
                case GPMODE_AUDIO_CTL_PAGE:
                    cpu_to_ube16(&buf[0], 24 + 6);
                    buf[2] = 0x70;
                    buf[3] = 0;
                    buf[4] = 0;
                    buf[5] = 0;
                    buf[6] = 0;
                    buf[7] = 0;
                    /* Fill with CDROM audio volume */
                    buf[17] = 0;
                    buf[19] = 0;
                    buf[21] = 0;
                    buf[23] = 0;
                    ide_atapi_cmd_reply(s, 24, max_len);
                case GPMODE_CAPABILITIES_PAGE:
                    cpu_to_ube16(&buf[0], 28 + 6);
                    buf[2] = 0x70;
                    buf[3] = 0;
                    buf[4] = 0;
                    buf[5] = 0;
                    buf[6] = 0;
                    buf[7] = 0;
                    buf[8] = 0x2a;
                    buf[9] = 0x12;
                    buf[10] = 0x00;
                    buf[11] = 0x00;
                    /* Claim PLAY_AUDIO capability (0x01) since some Linux
                       code checks for this to automount media. */
                    buf[12] = 0x71;
                    buf[13] = 3 << 5;
                    buf[14] = (1 << 0) | (1 << 3) | (1 << 5);
                    if (bdrv_is_locked(s->bs))
                        buf[6] |= 1 << 1;
                    buf[15] = 0x00;
                    cpu_to_ube16(&buf[16], 706);
                    buf[18] = 0;
                    buf[19] = 2;
                    cpu_to_ube16(&buf[20], 512);
                    cpu_to_ube16(&buf[22], 706);
                    buf[24] = 0;
                    buf[25] = 0;
                    buf[26] = 0;
                    buf[27] = 0;
                    ide_atapi_cmd_reply(s, 28, max_len);
                default:
                    goto error_cmd;
            case 1: /* changeable values */
                goto error_cmd;
            case 2: /* default values */
                goto error_cmd;
            default:
            case 3: /* saved values */
                                    ASC_SAVING_PARAMETERS_NOT_SUPPORTED);
    case GPCMD_REQUEST_SENSE:
        max_len = packet[4];
        memset(buf, 0, 18);
        buf[0] = 0x70 | (1 << 7);
        buf[2] = s->sense_key;
        buf[7] = 10;
        buf[12] = s->asc;
        if (s->sense_key == SENSE_UNIT_ATTENTION)
            s->sense_key = SENSE_NONE;
        ide_atapi_cmd_reply(s, 18, max_len);
    case GPCMD_PREVENT_ALLOW_MEDIUM_REMOVAL:
        if (bdrv_is_inserted(s->bs)) {
            bdrv_set_locked(s->bs, packet[4] & 1);
            ide_atapi_cmd_ok(s);
        } else {
            ide_atapi_cmd_error(s, SENSE_NOT_READY,
                                ASC_MEDIUM_NOT_PRESENT);
    case GPCMD_READ_10:
    case GPCMD_READ_12:
        {
            int nb_sectors, lba;
            if (packet[0] == GPCMD_READ_10)
                nb_sectors = ube16_to_cpu(packet + 7);
            else
                nb_sectors = ube32_to_cpu(packet + 6);
            lba = ube32_to_cpu(packet + 2);
            if (nb_sectors == 0) {
                ide_atapi_cmd_ok(s);
            ide_atapi_cmd_read(s, lba, nb_sectors, 2048);
    case GPCMD_READ_CD:
        {
            int nb_sectors, lba, transfer_request;
            nb_sectors = (packet[6] << 16) | (packet[7] << 8) | packet[8];
            lba = ube32_to_cpu(packet + 2);
            if (nb_sectors == 0) {
                ide_atapi_cmd_ok(s);
            transfer_request = packet[9];
            switch(transfer_request & 0xf8) {
            case 0x00:
                /* nothing */
                ide_atapi_cmd_ok(s);
            case 0x10:
                /* normal read */
                ide_atapi_cmd_read(s, lba, nb_sectors, 2048);
            case 0xf8:
                /* read all data */
                ide_atapi_cmd_read(s, lba, nb_sectors, 2352);
            default:
    case GPCMD_SEEK:
        {
            unsigned int lba;
            uint64_t total_sectors;
            bdrv_get_geometry(s->bs, &total_sectors);
            total_sectors >>= 2;
            if (total_sectors == 0) {
                ide_atapi_cmd_error(s, SENSE_NOT_READY,
                                    ASC_MEDIUM_NOT_PRESENT);
            lba = ube32_to_cpu(packet + 2);
            if (lba >= total_sectors) {
                                    ASC_LOGICAL_BLOCK_OOR);
            ide_atapi_cmd_ok(s);
    case GPCMD_START_STOP_UNIT:
        {
            int start, eject, err = 0;
            start = packet[4] & 1;
            eject = (packet[4] >> 1) & 1;
            if (eject) {
                err = bdrv_eject(s->bs, !start);
            switch (err) {
            case 0:
                ide_atapi_cmd_ok(s);
            case -EBUSY:
                ide_atapi_cmd_error(s, SENSE_NOT_READY,
                                    ASC_MEDIA_REMOVAL_PREVENTED);
            default:
                ide_atapi_cmd_error(s, SENSE_NOT_READY,
                                    ASC_MEDIUM_NOT_PRESENT);
    case GPCMD_MECHANISM_STATUS:
        {
            max_len = ube16_to_cpu(packet + 8);
            cpu_to_ube16(buf, 0);
            /* no current LBA */
            buf[2] = 0;
            buf[3] = 0;
            buf[4] = 0;
            buf[5] = 1;
            cpu_to_ube16(buf + 6, 0);
            ide_atapi_cmd_reply(s, 8, max_len);
    case GPCMD_READ_TOC_PMA_ATIP:
        {
            int format, msf, start_track, len;
            uint64_t total_sectors;
            bdrv_get_geometry(s->bs, &total_sectors);
            total_sectors >>= 2;
            if (total_sectors == 0) {
                ide_atapi_cmd_error(s, SENSE_NOT_READY,
                                    ASC_MEDIUM_NOT_PRESENT);
            format = packet[9] >> 6;
            msf = (packet[1] >> 1) & 1;
            start_track = packet[6];
            switch(format) {
            case 0:
                len = cdrom_read_toc(total_sectors, buf, msf, start_track);
                if (len < 0)
                    goto error_cmd;
                ide_atapi_cmd_reply(s, len, max_len);
            case 1:
                /* multi session : only a single session defined */
                memset(buf, 0, 12);
                buf[1] = 0x0a;
                buf[2] = 0x01;
                buf[3] = 0x01;
                ide_atapi_cmd_reply(s, 12, max_len);
            case 2:
                len = cdrom_read_toc_raw(total_sectors, buf, msf, start_track);
                if (len < 0)
                    goto error_cmd;
                ide_atapi_cmd_reply(s, len, max_len);
            default:
            error_cmd:
    case GPCMD_READ_CDVD_CAPACITY:
        {
            uint64_t total_sectors;
            bdrv_get_geometry(s->bs, &total_sectors);
            total_sectors >>= 2;
            if (total_sectors == 0) {
                ide_atapi_cmd_error(s, SENSE_NOT_READY,
                                    ASC_MEDIUM_NOT_PRESENT);
            /* NOTE: it is really the number of sectors minus 1 */
            cpu_to_ube32(buf, total_sectors - 1);
            cpu_to_ube32(buf + 4, 2048);
            ide_atapi_cmd_reply(s, 8, 8);
    case GPCMD_READ_DVD_STRUCTURE:
        {
            int media = packet[1];
            int format = packet[7];
            int ret;
            max_len = ube16_to_cpu(packet + 8);
            if (format < 0xff) {
                if (media_is_cd(s)) {
                                        ASC_INCOMPATIBLE_FORMAT);
                } else if (!media_present(s)) {
            memset(buf, 0, max_len > IDE_DMA_BUF_SECTORS * 512 + 4 ?
                   IDE_DMA_BUF_SECTORS * 512 + 4 : max_len);
            switch (format) {
                case 0x00 ... 0x7f:
                case 0xff:
                    if (media == 0) {
                        ret = ide_dvd_read_structure(s, format, packet, buf);
                        if (ret < 0)
                            ide_atapi_cmd_error(s, SENSE_ILLEGAL_REQUEST, -ret);
                        else
                            ide_atapi_cmd_reply(s, ret, max_len);
                    /* TODO: BD support, fall through for now */
                /* Generic disk structures */
                case 0x80: /* TODO: AACS volume identifier */
                case 0x81: /* TODO: AACS media serial number */
                case 0x82: /* TODO: AACS media identifier */
                case 0x83: /* TODO: AACS media key block */
                case 0x90: /* TODO: List of recognized format layers */
                case 0xc0: /* TODO: Write protection status */
                default:
    case GPCMD_SET_SPEED:
        ide_atapi_cmd_ok(s);
    case GPCMD_INQUIRY:
        max_len = packet[4];
        buf[0] = 0x05; /* CD-ROM */
        buf[1] = 0x80; /* removable */
        buf[2] = 0x00; /* ISO */
        buf[3] = 0x21; /* ATAPI-2 (XXX: put ATAPI-4 ?) */
        buf[4] = 31; /* additional length */
        buf[5] = 0; /* reserved */
        buf[6] = 0; /* reserved */
        buf[7] = 0; /* reserved */
        padstr8(buf + 8, 8, "QEMU");
        padstr8(buf + 16, 16, "QEMU DVD-ROM");
        padstr8(buf + 32, 4, s->version);
        ide_atapi_cmd_reply(s, 36, max_len);
    case GPCMD_GET_CONFIGURATION:
        {
            uint32_t len;
            uint8_t index = 0;
            /* only feature 0 is supported */
            if (packet[2] != 0 || packet[3] != 0) {
            /* XXX: could result in alignment problems in some architectures */
            /*
             * XXX: avoid overflow for io_buffer if max_len is bigger than
             *      the size of that buffer (dimensioned to max number of
             *      sectors to transfer at once)
             *
             *      Only a problem if the feature/profiles grow.
             */
            if (max_len > 512) /* XXX: assume 1 sector */
                max_len = 512;
            memset(buf, 0, max_len);
            /* 
             * the number of sectors from the media tells us which profile
             * to use as current.  0 means there is no media
             */
            if (media_is_dvd(s))
                cpu_to_ube16(buf + 6, MMC_PROFILE_DVD_ROM);
            else if (media_is_cd(s))
                cpu_to_ube16(buf + 6, MMC_PROFILE_CD_ROM);
            buf[10] = 0x02 | 0x01; /* persistent and current */
            len = 12; /* headers: 8 + 4 */
            len += ide_atapi_set_profile(buf, &index, MMC_PROFILE_DVD_ROM);
            len += ide_atapi_set_profile(buf, &index, MMC_PROFILE_CD_ROM);
            cpu_to_ube32(buf, len - 4); /* data length */
            ide_atapi_cmd_reply(s, len, max_len);
    default:
                            ASC_ILLEGAL_OPCODE);