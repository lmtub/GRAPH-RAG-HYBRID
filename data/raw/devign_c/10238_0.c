static void ide_ioport_write(void *opaque, uint32_t addr, uint32_t val)

{

    IDEState *ide_if = opaque;

    IDEState *s;

    int unit, n;

    int lba48 = 0;



#ifdef DEBUG_IDE

    printf("IDE: write addr=0x%x val=0x%02x\n", addr, val);

#endif



    addr &= 7;



    /* ignore writes to command block while busy with previous command */

    if (addr != 7 && (ide_if->cur_drive->status & (BUSY_STAT|DRQ_STAT)))

        return;



    switch(addr) {

    case 0:

        break;

    case 1:

	ide_clear_hob(ide_if);

        /* NOTE: data is written to the two drives */

	ide_if[0].hob_feature = ide_if[0].feature;

	ide_if[1].hob_feature = ide_if[1].feature;

        ide_if[0].feature = val;

        ide_if[1].feature = val;

        break;

    case 2:

	ide_clear_hob(ide_if);

	ide_if[0].hob_nsector = ide_if[0].nsector;

	ide_if[1].hob_nsector = ide_if[1].nsector;

        ide_if[0].nsector = val;

        ide_if[1].nsector = val;

        break;

    case 3:

	ide_clear_hob(ide_if);

	ide_if[0].hob_sector = ide_if[0].sector;

	ide_if[1].hob_sector = ide_if[1].sector;

        ide_if[0].sector = val;

        ide_if[1].sector = val;

        break;

    case 4:

	ide_clear_hob(ide_if);

	ide_if[0].hob_lcyl = ide_if[0].lcyl;

	ide_if[1].hob_lcyl = ide_if[1].lcyl;

        ide_if[0].lcyl = val;

        ide_if[1].lcyl = val;

        break;

    case 5:

	ide_clear_hob(ide_if);

	ide_if[0].hob_hcyl = ide_if[0].hcyl;

	ide_if[1].hob_hcyl = ide_if[1].hcyl;

        ide_if[0].hcyl = val;

        ide_if[1].hcyl = val;

        break;

    case 6:

	/* FIXME: HOB readback uses bit 7 */

        ide_if[0].select = (val & ~0x10) | 0xa0;

        ide_if[1].select = (val | 0x10) | 0xa0;

        /* select drive */

        unit = (val >> 4) & 1;

        s = ide_if + unit;

        ide_if->cur_drive = s;

        break;

    default:

    case 7:

        /* command */

#if defined(DEBUG_IDE)

        printf("ide: CMD=%02x\n", val);

#endif

        s = ide_if->cur_drive;

        /* ignore commands to non existant slave */

        if (s != ide_if && !s->bs)

            break;



        /* Only DEVICE RESET is allowed while BSY or/and DRQ are set */

        if ((s->status & (BUSY_STAT|DRQ_STAT)) && val != WIN_DEVICE_RESET)

            break;



        switch(val) {

        case WIN_IDENTIFY:

            if (s->bs && !s->is_cdrom) {

                if (!s->is_cf)

                    ide_identify(s);

                else

                    ide_cfata_identify(s);

                s->status = READY_STAT | SEEK_STAT;

                ide_transfer_start(s, s->io_buffer, 512, ide_transfer_stop);

            } else {

                if (s->is_cdrom) {

                    ide_set_signature(s);

                }

                ide_abort_command(s);

            }

            ide_set_irq(s);

            break;

        case WIN_SPECIFY:

        case WIN_RECAL:

            s->error = 0;

            s->status = READY_STAT | SEEK_STAT;

            ide_set_irq(s);

            break;

        case WIN_SETMULT:

            if (s->is_cf && s->nsector == 0) {

                /* Disable Read and Write Multiple */

                s->mult_sectors = 0;

                s->status = READY_STAT | SEEK_STAT;

            } else if ((s->nsector & 0xff) != 0 &&

                ((s->nsector & 0xff) > MAX_MULT_SECTORS ||

                 (s->nsector & (s->nsector - 1)) != 0)) {

                ide_abort_command(s);

            } else {

                s->mult_sectors = s->nsector & 0xff;

                s->status = READY_STAT | SEEK_STAT;

            }

            ide_set_irq(s);

            break;

        case WIN_VERIFY_EXT:

	    lba48 = 1;

        case WIN_VERIFY:

        case WIN_VERIFY_ONCE:

            /* do sector number check ? */

	    ide_cmd_lba48_transform(s, lba48);

            s->status = READY_STAT | SEEK_STAT;

            ide_set_irq(s);

            break;

	case WIN_READ_EXT:

	    lba48 = 1;

        case WIN_READ:

        case WIN_READ_ONCE:

            if (!s->bs)

                goto abort_cmd;

	    ide_cmd_lba48_transform(s, lba48);

            s->req_nb_sectors = 1;

            ide_sector_read(s);

            break;

	case WIN_WRITE_EXT:

	    lba48 = 1;

        case WIN_WRITE:

        case WIN_WRITE_ONCE:

        case CFA_WRITE_SECT_WO_ERASE:

        case WIN_WRITE_VERIFY:

	    ide_cmd_lba48_transform(s, lba48);

            s->error = 0;

            s->status = SEEK_STAT | READY_STAT;

            s->req_nb_sectors = 1;

            ide_transfer_start(s, s->io_buffer, 512, ide_sector_write);

            s->media_changed = 1;

            break;

	case WIN_MULTREAD_EXT:

	    lba48 = 1;

        case WIN_MULTREAD:

            if (!s->mult_sectors)

                goto abort_cmd;

	    ide_cmd_lba48_transform(s, lba48);

            s->req_nb_sectors = s->mult_sectors;

            ide_sector_read(s);

            break;

        case WIN_MULTWRITE_EXT:

	    lba48 = 1;

        case WIN_MULTWRITE:

        case CFA_WRITE_MULTI_WO_ERASE:

            if (!s->mult_sectors)

                goto abort_cmd;

	    ide_cmd_lba48_transform(s, lba48);

            s->error = 0;

            s->status = SEEK_STAT | READY_STAT;

            s->req_nb_sectors = s->mult_sectors;

            n = s->nsector;

            if (n > s->req_nb_sectors)

                n = s->req_nb_sectors;

            ide_transfer_start(s, s->io_buffer, 512 * n, ide_sector_write);

            s->media_changed = 1;

            break;

	case WIN_READDMA_EXT:

	    lba48 = 1;

        case WIN_READDMA:

        case WIN_READDMA_ONCE:

            if (!s->bs)

                goto abort_cmd;

	    ide_cmd_lba48_transform(s, lba48);

            ide_sector_read_dma(s);

            break;

	case WIN_WRITEDMA_EXT:

	    lba48 = 1;

        case WIN_WRITEDMA:

        case WIN_WRITEDMA_ONCE:

            if (!s->bs)

                goto abort_cmd;

	    ide_cmd_lba48_transform(s, lba48);

            ide_sector_write_dma(s);

            s->media_changed = 1;

            break;

        case WIN_READ_NATIVE_MAX_EXT:

	    lba48 = 1;

        case WIN_READ_NATIVE_MAX:

	    ide_cmd_lba48_transform(s, lba48);

            ide_set_sector(s, s->nb_sectors - 1);

            s->status = READY_STAT | SEEK_STAT;

            ide_set_irq(s);

            break;

        case WIN_CHECKPOWERMODE1:

        case WIN_CHECKPOWERMODE2:

            s->nsector = 0xff; /* device active or idle */

            s->status = READY_STAT | SEEK_STAT;

            ide_set_irq(s);

            break;

        case WIN_SETFEATURES:

            if (!s->bs)

                goto abort_cmd;

            /* XXX: valid for CDROM ? */

            switch(s->feature) {

            case 0xcc: /* reverting to power-on defaults enable */

            case 0x66: /* reverting to power-on defaults disable */

            case 0x02: /* write cache enable */

            case 0x82: /* write cache disable */

            case 0xaa: /* read look-ahead enable */

            case 0x55: /* read look-ahead disable */

            case 0x05: /* set advanced power management mode */

            case 0x85: /* disable advanced power management mode */

            case 0x69: /* NOP */

            case 0x67: /* NOP */

            case 0x96: /* NOP */

            case 0x9a: /* NOP */

            case 0x42: /* enable Automatic Acoustic Mode */

            case 0xc2: /* disable Automatic Acoustic Mode */

                s->status = READY_STAT | SEEK_STAT;

                ide_set_irq(s);

                break;

            case 0x03: { /* set transfer mode */

		uint8_t val = s->nsector & 0x07;



		switch (s->nsector >> 3) {

		    case 0x00: /* pio default */

		    case 0x01: /* pio mode */

			put_le16(s->identify_data + 62,0x07);

			put_le16(s->identify_data + 63,0x07);

			put_le16(s->identify_data + 88,0x3f);

			break;

                    case 0x02: /* sigle word dma mode*/

			put_le16(s->identify_data + 62,0x07 | (1 << (val + 8)));

			put_le16(s->identify_data + 63,0x07);

			put_le16(s->identify_data + 88,0x3f);

			break;

		    case 0x04: /* mdma mode */

			put_le16(s->identify_data + 62,0x07);

			put_le16(s->identify_data + 63,0x07 | (1 << (val + 8)));

			put_le16(s->identify_data + 88,0x3f);

			break;

		    case 0x08: /* udma mode */

			put_le16(s->identify_data + 62,0x07);

			put_le16(s->identify_data + 63,0x07);

			put_le16(s->identify_data + 88,0x3f | (1 << (val + 8)));

			break;

		    default:

			goto abort_cmd;

		}

                s->status = READY_STAT | SEEK_STAT;

                ide_set_irq(s);

                break;

	    }

            default:

                goto abort_cmd;

            }

            break;

        case WIN_FLUSH_CACHE:

        case WIN_FLUSH_CACHE_EXT:

            if (s->bs)

                bdrv_flush(s->bs);

	    s->status = READY_STAT | SEEK_STAT;

            ide_set_irq(s);

            break;

        case WIN_STANDBY:

        case WIN_STANDBY2:

        case WIN_STANDBYNOW1:

        case WIN_STANDBYNOW2:

        case WIN_IDLEIMMEDIATE:

        case CFA_IDLEIMMEDIATE:

        case WIN_SETIDLE1:

        case WIN_SETIDLE2:

        case WIN_SLEEPNOW1:

        case WIN_SLEEPNOW2:

            s->status = READY_STAT;

            ide_set_irq(s);

            break;

            /* ATAPI commands */

        case WIN_PIDENTIFY:

            if (s->is_cdrom) {

                ide_atapi_identify(s);

                s->status = READY_STAT | SEEK_STAT;

                ide_transfer_start(s, s->io_buffer, 512, ide_transfer_stop);

            } else {

                ide_abort_command(s);

            }

            ide_set_irq(s);

            break;

        case WIN_DIAGNOSE:

            ide_set_signature(s);

            s->status = READY_STAT | SEEK_STAT;

            s->error = 0x01;

            ide_set_irq(s);

            break;

        case WIN_SRST:

            if (!s->is_cdrom)

                goto abort_cmd;

            ide_set_signature(s);

            s->status = 0x00; /* NOTE: READY is _not_ set */

            s->error = 0x01;

            break;

        case WIN_PACKETCMD:

            if (!s->is_cdrom)

                goto abort_cmd;

            /* overlapping commands not supported */

            if (s->feature & 0x02)

                goto abort_cmd;

            s->status = READY_STAT | SEEK_STAT;

            s->atapi_dma = s->feature & 1;

            s->nsector = 1;

            ide_transfer_start(s, s->io_buffer, ATAPI_PACKET_SIZE,

                               ide_atapi_cmd);

            break;

        /* CF-ATA commands */

        case CFA_REQ_EXT_ERROR_CODE:

            if (!s->is_cf)

                goto abort_cmd;

            s->error = 0x09;    /* miscellaneous error */

            s->status = READY_STAT | SEEK_STAT;

            ide_set_irq(s);

            break;

        case CFA_ERASE_SECTORS:

        case CFA_WEAR_LEVEL:

            if (!s->is_cf)

                goto abort_cmd;

            if (val == CFA_WEAR_LEVEL)

                s->nsector = 0;

            if (val == CFA_ERASE_SECTORS)

                s->media_changed = 1;

            s->error = 0x00;

            s->status = READY_STAT | SEEK_STAT;

            ide_set_irq(s);

            break;

        case CFA_TRANSLATE_SECTOR:

            if (!s->is_cf)

                goto abort_cmd;

            s->error = 0x00;

            s->status = READY_STAT | SEEK_STAT;

            memset(s->io_buffer, 0, 0x200);

            s->io_buffer[0x00] = s->hcyl;			/* Cyl MSB */

            s->io_buffer[0x01] = s->lcyl;			/* Cyl LSB */

            s->io_buffer[0x02] = s->select;			/* Head */

            s->io_buffer[0x03] = s->sector;			/* Sector */

            s->io_buffer[0x04] = ide_get_sector(s) >> 16;	/* LBA MSB */

            s->io_buffer[0x05] = ide_get_sector(s) >> 8;	/* LBA */

            s->io_buffer[0x06] = ide_get_sector(s) >> 0;	/* LBA LSB */

            s->io_buffer[0x13] = 0x00;				/* Erase flag */

            s->io_buffer[0x18] = 0x00;				/* Hot count */

            s->io_buffer[0x19] = 0x00;				/* Hot count */

            s->io_buffer[0x1a] = 0x01;				/* Hot count */

            ide_transfer_start(s, s->io_buffer, 0x200, ide_transfer_stop);

            ide_set_irq(s);

            break;

        case CFA_ACCESS_METADATA_STORAGE:

            if (!s->is_cf)

                goto abort_cmd;

            switch (s->feature) {

            case 0x02:	/* Inquiry Metadata Storage */

                ide_cfata_metadata_inquiry(s);

                break;

            case 0x03:	/* Read Metadata Storage */

                ide_cfata_metadata_read(s);

                break;

            case 0x04:	/* Write Metadata Storage */

                ide_cfata_metadata_write(s);

                break;

            default:

                goto abort_cmd;

            }

            ide_transfer_start(s, s->io_buffer, 0x200, ide_transfer_stop);

            s->status = 0x00; /* NOTE: READY is _not_ set */

            ide_set_irq(s);

            break;

        case IBM_SENSE_CONDITION:

            if (!s->is_cf)

                goto abort_cmd;

            switch (s->feature) {

            case 0x01:  /* sense temperature in device */

                s->nsector = 0x50;      /* +20 C */

                break;

            default:

                goto abort_cmd;

            }

            s->status = READY_STAT | SEEK_STAT;

            ide_set_irq(s);

            break;

        default:

        abort_cmd:

            ide_abort_command(s);

            ide_set_irq(s);

            break;

        }

    }

}
